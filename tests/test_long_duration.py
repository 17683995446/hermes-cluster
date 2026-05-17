"""
Hermes Cluster - 长时间压力测试框架
模拟24小时连续运行，检测内存泄漏、性能退化、资源耗尽等问题
"""

import asyncio
import gc
import logging
import os
import psutil
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable

import pytest

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hermes_cluster.geo.region import (
    CrossRegionOrchestrator, RegionInfo, RegionStatus,
    DataPartition, SyncMode, RoutingStrategy
)
from hermes_cluster.scale.pool import (
    AgentPoolManager, AutoScaler, HierarchicalScheduler,
    AgentSpec, AgentState, ScalingPolicy, PerformanceBenchmark
)
from hermes_cluster.lowcode.editor import (
    WorkflowEditor, WorkflowNode, NodeType, Connection,
    WorkflowExecutor, WorkflowValidator
)
from hermes_cluster.models.router import (
    ModelRouter, ChatRequest, ChatMessage, RouteStrategy
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """内存快照"""
    timestamp: datetime
    rss_mb: float
    vms_mb: float
    num_objects: int
    tracemalloc_size: int
    tracemalloc_peak: int


@dataclass
class PerformanceMetric:
    """性能指标"""
    timestamp: datetime
    operation: str
    latency_ms: float
    throughput: float
    error_rate: float


@dataclass
class LongDurationReport:
    """长时间运行报告"""
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_hours: float = 0.0
    memory_snapshots: List[MemorySnapshot] = field(default_factory=list)
    performance_metrics: List[PerformanceMetric] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    memory_leak_detected: bool = False
    performance_degradation: bool = False
    errors: List[str] = field(default_factory=list)


class LongDurationTester:
    """长时间运行测试器"""

    def __init__(self, duration_hours: float = 24.0, snapshot_interval: int = 60):
        self.duration_hours = duration_hours
        self.snapshot_interval = snapshot_interval  # 秒
        self.report = LongDurationReport(start_time=datetime.now())
        self._stop_event = asyncio.Event()
        self._baseline_memory: Optional[MemorySnapshot] = None
        self._baseline_performance: Optional[PerformanceMetric] = None
        self._process = psutil.Process()

    async def run(self) -> LongDurationReport:
        """运行长时间测试"""
        logger.info(f"开始长时间压力测试，计划运行 {self.duration_hours} 小时")

        # 启动内存追踪
        tracemalloc.start()

        # 获取基线
        await self._capture_baseline()

        # 启动监控任务
        monitor_task = asyncio.create_task(self._monitor_loop())

        # 启动压力测试任务
        stress_tasks = [
            asyncio.create_task(self._stress_agent_pool()),
            asyncio.create_task(self._stress_workflow_execution()),
            asyncio.create_task(self._stress_geo_routing()),
            asyncio.create_task(self._stress_model_routing()),
        ]

        # 等待指定时间或停止信号
        try:
            await asyncio.wait_for(
                self._stop_event.wait(),
                timeout=self.duration_hours * 3600
            )
        except asyncio.TimeoutError:
            logger.info("达到测试时长，正常结束")

        # 停止所有任务
        for task in stress_tasks:
            task.cancel()
        monitor_task.cancel()

        # 生成报告
        self.report.end_time = datetime.now()
        self.report.duration_hours = (
            self.report.end_time - self.report.start_time
        ).total_seconds() / 3600

        self._analyze_results()

        tracemalloc.stop()
        return self.report

    async def _capture_baseline(self):
        """捕获基线数据"""
        self._baseline_memory = await self._take_memory_snapshot()
        logger.info(f"内存基线: RSS={self._baseline_memory.rss_mb:.2f}MB")

    async def _take_memory_snapshot(self) -> MemorySnapshot:
        """获取内存快照"""
        # 强制垃圾回收
        gc.collect()

        mem_info = self._process.memory_info()
        current, peak = tracemalloc.get_traced_memory()

        return MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=mem_info.rss / 1024 / 1024,
            vms_mb=mem_info.vms / 1024 / 1024,
            num_objects=len(gc.get_objects()),
            tracemalloc_size=current,
            tracemalloc_peak=peak
        )

    async def _monitor_loop(self):
        """监控循环"""
        while not self._stop_event.is_set():
            try:
                snapshot = await self._take_memory_snapshot()
                self.report.memory_snapshots.append(snapshot)

                # 检测内存泄漏
                if self._baseline_memory:
                    memory_growth = snapshot.rss_mb - self._baseline_memory.rss_mb
                    if memory_growth > 100:  # 超过100MB增长
                        self._record_anomaly(
                            "memory_growth",
                            f"内存增长 {memory_growth:.2f}MB",
                            snapshot
                        )

                # 限制快照数量，避免内存溢出
                if len(self.report.memory_snapshots) > 1000:
                    self.report.memory_snapshots = (
                        self.report.memory_snapshots[:100] +
                        self.report.memory_snapshots[-900:]
                    )

            except Exception as e:
                logger.error(f"监控循环错误: {e}")

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.snapshot_interval
                )
            except asyncio.TimeoutError:
                pass

    async def _stress_agent_pool(self):
        """Agent池压力测试"""
        manager = AgentPoolManager()

        try:
            # 创建多个池
            pools = []
            for i in range(5):
                pool = await manager.create_pool(
                    name=f"stress_pool_{i}",
                    agent_type="stress_agent",
                    min_size=5,
                    max_size=50,
                    initial_size=10
                )
                pools.append(pool)

            iteration = 0
            while not self._stop_event.is_set():
                iteration += 1
                start_time = time.time()

                try:
                    # 随机获取和释放agent
                    tasks = []
                    for pool in pools:
                        for _ in range(10):
                            tasks.append(manager.acquire_agent(pool.pool_id))

                    agents = await asyncio.gather(*tasks, return_exceptions=True)
                    acquired = [a for a in agents if not isinstance(a, Exception) and a is not None]

                    # 模拟工作
                    await asyncio.sleep(0.01)

                    # 释放agent
                    for agent in acquired:
                        await manager.release_agent(agent.agent_id)

                    # 动态扩缩容
                    if iteration % 100 == 0:
                        for pool in pools:
                            new_size = (iteration % 40) + 10
                            await manager.scale_pool(pool.pool_id, new_size)

                    # 记录性能指标
                    latency = (time.time() - start_time) * 1000
                    self._record_performance("agent_pool", latency, len(acquired))

                except Exception as e:
                    self.report.errors.append(f"Agent池压力测试错误: {e}")

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("Agent池压力测试已停止")

    async def _stress_workflow_execution(self):
        """工作流执行压力测试"""
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)

        try:
            # 创建复杂工作流
            workflow_id = editor.create_workflow("stress_workflow")

            # 添加节点
            nodes = []
            for i in range(20):
                node = WorkflowNode(
                    node_id=f"node_{i}",
                    node_type=NodeType.AGENT if i % 3 == 0 else NodeType.FUNCTION,
                    name=f"Stress Node {i}",
                    config={"timeout": 1.0, "retry": 2}
                )
                editor.add_node(workflow_id, node)
                nodes.append(node)

            # 连接节点
            for i in range(len(nodes) - 1):
                editor.connect_nodes(
                    workflow_id,
                    nodes[i].node_id,
                    nodes[i + 1].node_id
                )

            iteration = 0
            while not self._stop_event.is_set():
                iteration += 1
                start_time = time.time()

                try:
                    # 执行工作流
                    context = {"iteration": iteration, "data": "x" * 1000}
                    await executor.execute(workflow_id, context)

                    latency = (time.time() - start_time) * 1000
                    self._record_performance("workflow", latency, 1)

                except Exception as e:
                    self.report.errors.append(f"工作流压力测试错误: {e}")

                await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            logger.info("工作流压力测试已停止")

    async def _stress_geo_routing(self):
        """地理路由压力测试"""
        orchestrator = CrossRegionOrchestrator()
        region_manager = orchestrator.get_region_manager()

        try:
            # 注册多个区域
            regions = []
            for i in range(10):
                region = RegionInfo(
                    region_id=f"region_{i}",
                    name=f"Region {i}",
                    location=f"location_{i % 5}",
                    timezone="UTC",
                    endpoint=f"http://region{i}.example.com",
                    status=RegionStatus.ACTIVE,
                    capacity=1000,
                    current_load=0
                )
                await region_manager.register_region(region)
                regions.append(region)

            iteration = 0
            while not self._stop_event.is_set():
                iteration += 1
                start_time = time.time()

                try:
                    # 随机更新区域状态
                    for region in regions:
                        if iteration % 50 == 0:
                            load = (iteration * 10) % region.capacity
                            await region_manager.update_region_status(
                                region.region_id,
                                RegionStatus.ACTIVE,
                                load
                            )

                    # 路由请求
                    for i in range(20):
                        await region_manager.select_region(
                            client_location=f"location_{i % 5}"
                        )

                    latency = (time.time() - start_time) * 1000
                    self._record_performance("geo_routing", latency, 20)

                except Exception as e:
                    self.report.errors.append(f"地理路由压力测试错误: {e}")

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("地理路由压力测试已停止")

    async def _stress_model_routing(self):
        """模型路由压力测试"""
        router = ModelRouter()

        try:
            # 注册模拟提供商和模型
            from hermes_cluster.models.router import (
                ProviderConfig, ProviderStatus, ModelInfo,
                ModelCapability, OpenAICompatibleProvider
            )

            config = ProviderConfig(
                provider_id="mock_provider",
                name="Mock Provider",
                base_url="http://mock.example.com",
                api_key="mock_key",
                status=ProviderStatus.ACTIVE
            )
            provider = OpenAICompatibleProvider(config)
            router.register_provider(provider)

            for i in range(5):
                model = ModelInfo(
                    model_id=f"model_{i}",
                    provider_id="mock_provider",
                    display_name=f"Model {i}",
                    capabilities={ModelCapability.CHAT},
                    max_context_tokens=4096,
                    max_output_tokens=1024
                )
                router.register_model(model)

            iteration = 0
            while not self._stop_event.is_set():
                iteration += 1
                start_time = time.time()

                try:
                    # 路由请求
                    for i in range(10):
                        await router.route(model=f"model_{i % 5}")

                    latency = (time.time() - start_time) * 1000
                    self._record_performance("model_routing", latency, 10)

                except Exception as e:
                    self.report.errors.append(f"模型路由压力测试错误: {e}")

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("模型路由压力测试已停止")

    def _record_performance(self, operation: str, latency_ms: float, throughput: float):
        """记录性能指标"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            operation=operation,
            latency_ms=latency_ms,
            throughput=throughput,
            error_rate=0.0
        )
        self.report.performance_metrics.append(metric)

        # 检测性能退化
        if self._baseline_performance and operation == self._baseline_performance.operation:
            if latency_ms > self._baseline_performance.latency_ms * 2:
                self._record_anomaly(
                    "performance_degradation",
                    f"{operation} 延迟退化: {latency_ms:.2f}ms",
                    metric
                )

        # 限制指标数量
        if len(self.report.performance_metrics) > 10000:
            self.report.performance_metrics = self.report.performance_metrics[-5000:]

    def _record_anomaly(self, anomaly_type: str, message: str, data: Any):
        """记录异常"""
        anomaly = {
            "type": anomaly_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": str(data)[:200]
        }
        self.report.anomalies.append(anomaly)
        logger.warning(f"检测到异常 [{anomaly_type}]: {message}")

    def _analyze_results(self):
        """分析测试结果"""
        if len(self.report.memory_snapshots) >= 2:
            first = self.report.memory_snapshots[0]
            last = self.report.memory_snapshots[-1]
            memory_growth = last.rss_mb - first.rss_mb

            # 检测内存泄漏：内存持续增长超过阈值
            if memory_growth > 50:  # 50MB
                self.report.memory_leak_detected = True
                logger.warning(f"检测到潜在内存泄漏: {memory_growth:.2f}MB")

        # 分析性能退化
        if self.report.performance_metrics:
            by_operation = defaultdict(list)
            for m in self.report.performance_metrics:
                by_operation[m.operation].append(m.latency_ms)

            for op, latencies in by_operation.items():
                if len(latencies) >= 10:
                    first_half = sum(latencies[:len(latencies)//2]) / (len(latencies)//2)
                    second_half = sum(latencies[len(latencies)//2:]) / (len(latencies) - len(latencies)//2)
                    if second_half > first_half * 1.5:
                        self.report.performance_degradation = True
                        logger.warning(f"{op} 检测到性能退化: {first_half:.2f}ms -> {second_half:.2f}ms")

    def stop(self):
        """停止测试"""
        self._stop_event.set()


# ============ 测试用例 ============

@pytest.mark.asyncio
async def test_long_duration_memory_stability():
    """测试长时间内存稳定性"""
    # 使用较短的测试时间进行CI测试
    tester = LongDurationTester(duration_hours=0.05, snapshot_interval=5)  # 3分钟测试

    # 设置提前停止
    async def early_stop():
        await asyncio.sleep(30)  # 30秒后停止
        tester.stop()

    asyncio.create_task(early_stop())

    report = await tester.run()

    # 验证
    assert report.duration_hours > 0
    assert len(report.memory_snapshots) > 0

    # 检查内存泄漏
    if report.memory_leak_detected:
        pytest.fail(f"检测到内存泄漏: {report.anomalies}")

    print(f"\n长时间测试完成:")
    print(f"  运行时长: {report.duration_hours:.2f} 小时")
    print(f"  内存快照数: {len(report.memory_snapshots)}")
    print(f"  性能指标数: {len(report.performance_metrics)}")
    print(f"  异常数: {len(report.anomalies)}")
    print(f"  错误数: {len(report.errors)}")


@pytest.mark.asyncio
async def test_agent_pool_long_running():
    """测试Agent池长时间运行"""
    manager = AgentPoolManager()

    # 创建池
    pool = await manager.create_pool(
        name="long_running_pool",
        agent_type="test_agent",
        min_size=5,
        max_size=20,
        initial_size=5
    )

    # 模拟长时间运行
    errors = []
    for i in range(100):  # 模拟100次循环
        try:
            # 获取agent
            agents = []
            for _ in range(3):
                agent = await manager.acquire_agent(pool.pool_id)
                if agent:
                    agents.append(agent)

            # 模拟工作
            await asyncio.sleep(0.01)

            # 释放agent
            for agent in agents:
                await manager.release_agent(agent.agent_id)

            # 扩缩容
            if i % 20 == 0:
                new_size = 10 + (i % 10)
                await manager.scale_pool(pool.pool_id, new_size)

        except Exception as e:
            errors.append(str(e))

    # 验证
    final_pool = (await manager.list_pools())[0]
    assert final_pool.current_size >= pool.min_size
    assert final_pool.current_size <= pool.max_size
    assert len(errors) == 0, f"出现错误: {errors}"


@pytest.mark.asyncio
async def test_workflow_long_running():
    """测试工作流长时间运行"""
    editor = WorkflowEditor()
    executor = WorkflowExecutor(editor)

    # 创建工作流
    workflow = editor.create_workflow("long_running_workflow")
    workflow_id = workflow.id

    # 添加节点
    for i in range(10):
        node = WorkflowNode(
            id=f"node_{i}",
            type=NodeType.FUNCTION,
            name=f"Node {i}",
            config={"code": f"context['val_{i}'] = {i}"}
        )
        editor.add_node(workflow_id, node)

    # 连接节点
    for i in range(9):
        editor.connect_nodes(workflow_id, f"node_{i}", f"node_{i+1}")

    # 多次执行
    errors = []
    for i in range(50):
        try:
            context = {"iteration": i}
            await executor.execute(workflow_id, context)
        except Exception as e:
            errors.append(str(e))

    assert len(errors) == 0, f"工作流执行错误: {errors}"


@pytest.mark.asyncio
async def test_geo_routing_long_running():
    """测试地理路由长时间运行"""
    orchestrator = CrossRegionOrchestrator()
    region_manager = orchestrator.get_region_manager()

    # 注册区域
    regions = []
    for i in range(5):
        region = RegionInfo(
            region_id=f"region_{i}",
            name=f"Region {i}",
            location=f"loc_{i}",
            timezone="UTC",
            endpoint=f"http://r{i}.example.com"
        )
        await region_manager.register_region(region)
        regions.append(region)

    # 长时间路由测试
    errors = []
    for i in range(200):
        try:
            # 更新负载
            for j, region in enumerate(regions):
                await region_manager.update_region_status(
                    region.region_id,
                    RegionStatus.ACTIVE,
                    load=(i * 10 + j * 50) % region.capacity
                )

            # 路由请求
            selected = await region_manager.select_region()
            assert selected is not None

        except Exception as e:
            errors.append(str(e))

    assert len(errors) == 0, f"地理路由错误: {errors}"


@pytest.mark.asyncio
async def test_memory_leak_detection():
    """测试内存泄漏检测机制"""
    tester = LongDurationTester(duration_hours=0.01, snapshot_interval=2)

    # 快速测试
    async def quick_stop():
        await asyncio.sleep(10)
        tester.stop()

    asyncio.create_task(quick_stop())
    report = await tester.run()

    # 验证报告结构
    assert report.start_time is not None
    assert len(report.memory_snapshots) >= 2

    # 分析内存趋势
    if len(report.memory_snapshots) >= 3:
        rss_values = [s.rss_mb for s in report.memory_snapshots]
        # 检查是否单调增长（潜在泄漏）
        increasing_count = sum(
            1 for i in range(1, len(rss_values))
            if rss_values[i] > rss_values[i-1]
        )
        growth_ratio = increasing_count / (len(rss_values) - 1)

        print(f"\n内存趋势分析:")
        print(f"  初始RSS: {rss_values[0]:.2f}MB")
        print(f"  最终RSS: {rss_values[-1]:.2f}MB")
        print(f"  增长比例: {growth_ratio:.2%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
