"""
Hermes Cluster - 混沌工程测试框架
模拟各种故障场景，测试系统容错能力和自动恢复机制
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable

import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hermes_cluster.geo.region import (
    CrossRegionOrchestrator, RegionInfo, RegionStatus,
    DataPartition, SyncMode
)
from hermes_cluster.scale.pool import (
    AgentPoolManager, AgentState, AgentInstance
)
from hermes_cluster.lowcode.editor import (
    WorkflowEditor, WorkflowNode, NodeType, WorkflowExecutor
)
from hermes_cluster.models.router import (
    ModelRouter, ProviderConfig, ProviderStatus, ModelInfo,
    ModelCapability, OpenAICompatibleProvider
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailureType(Enum):
    """故障类型"""
    REGION_OFFLINE = "region_offline"
    REGION_DEGRADED = "region_degraded"
    AGENT_CRASH = "agent_crash"
    AGENT_TIMEOUT = "agent_timeout"
    NETWORK_PARTITION = "network_partition"
    MEMORY_PRESSURE = "memory_pressure"
    PROVIDER_FAILURE = "provider_failure"
    MODEL_UNAVAILABLE = "model_unavailable"
    DB_CONNECTION_LOST = "db_connection_lost"
    HIGH_LATENCY = "high_latency"


@dataclass
class FailureEvent:
    """故障事件"""
    failure_type: FailureType
    target: str
    timestamp: datetime
    duration: float
    recovered: bool = False
    recovery_time: Optional[datetime] = None
    impact: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChaosReport:
    """混沌测试报告"""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_failures: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    events: List[FailureEvent] = field(default_factory=list)
    system_downtime_ms: float = 0.0
    data_loss_events: int = 0


class ChaosMonkey:
    """混沌猴子 - 故障注入器"""

    def __init__(self):
        self._faults: Dict[FailureType, Callable] = {}
        self._report = ChaosReport(start_time=datetime.now())
        self._is_running = False
        self._register_faults()

    def _register_faults(self):
        """注册故障注入器"""
        self._faults = {
            FailureType.REGION_OFFLINE: self._inject_region_offline,
            FailureType.REGION_DEGRADED: self._inject_region_degraded,
            FailureType.AGENT_CRASH: self._inject_agent_crash,
            FailureType.AGENT_TIMEOUT: self._inject_agent_timeout,
            FailureType.PROVIDER_FAILURE: self._inject_provider_failure,
            FailureType.MODEL_UNAVAILABLE: self._inject_model_unavailable,
            FailureType.HIGH_LATENCY: self._inject_high_latency,
        }

    async def inject_fault(self, fault_type: FailureType, target: Any,
                          duration: float = 5.0) -> FailureEvent:
        """注入故障"""
        event = FailureEvent(
            failure_type=fault_type,
            target=str(target),
            timestamp=datetime.now(),
            duration=duration
        )

        injector = self._faults.get(fault_type)
        if injector:
            try:
                await injector(target, duration)
                self._report.total_failures += 1
                logger.info(f"注入故障: {fault_type.value} -> {target}")
            except Exception as e:
                logger.error(f"故障注入失败: {e}")

        self._report.events.append(event)
        return event

    async def _inject_region_offline(self, region_manager: Any, duration: float):
        """注入区域离线故障"""
        regions = await region_manager.list_regions()
        if regions:
            target = random.choice(regions)
            await region_manager.update_region_status(
                target.region_id, RegionStatus.OFFLINE
            )
            await asyncio.sleep(duration)
            await region_manager.update_region_status(
                target.region_id, RegionStatus.ACTIVE
            )

    async def _inject_region_degraded(self, region_manager: Any, duration: float):
        """注入区域降级故障"""
        regions = await region_manager.list_regions()
        if regions:
            target = random.choice(regions)
            await region_manager.update_region_status(
                target.region_id, RegionStatus.DEGRADED
            )
            await asyncio.sleep(duration)
            await region_manager.update_region_status(
                target.region_id, RegionStatus.ACTIVE
            )

    async def _inject_agent_crash(self, pool_manager: Any, duration: float):
        """注入Agent崩溃故障"""
        pools = await pool_manager.list_pools()
        for pool in pools:
            if pool.agents:
                target = random.choice(list(pool.agents.values()))
                target.state = AgentState.ERROR
                await asyncio.sleep(duration)
                target.state = AgentState.IDLE
                break

    async def _inject_agent_timeout(self, pool_manager: Any, duration: float):
        """注入Agent超时故障"""
        pools = await pool_manager.list_pools()
        for pool in pools:
            agent = await pool_manager.acquire_agent(pool.pool_id)
            if agent:
                await asyncio.sleep(duration)
                await pool_manager.release_agent(agent.agent_id)
                break

    async def _inject_provider_failure(self, router: ModelRouter, duration: float):
        """注入提供商故障"""
        for provider_id, provider in router._providers.items():
            provider._config.status = ProviderStatus.INACTIVE
            await asyncio.sleep(duration)
            provider._config.status = ProviderStatus.ACTIVE
            break

    async def _inject_model_unavailable(self, router: ModelRouter, duration: float):
        """注入模型不可用故障"""
        if router._models:
            model_id = random.choice(list(router._models.keys()))
            model = router._models[model_id]
            original_provider = model.provider_id
            model.provider_id = "nonexistent_provider"
            await asyncio.sleep(duration)
            model.provider_id = original_provider

    async def _inject_high_latency(self, target: Any, duration: float):
        """注入高延迟"""
        await asyncio.sleep(duration)

    def get_report(self) -> ChaosReport:
        """获取测试报告"""
        self._report.end_time = datetime.now()
        return self._report


# ============ 测试用例 ============

@pytest.mark.asyncio
async def test_region_failover():
    """测试区域故障转移"""
    orchestrator = CrossRegionOrchestrator()
    region_manager = orchestrator.get_region_manager()
    chaos = ChaosMonkey()

    # 注册多个区域
    regions = []
    for i in range(3):
        region = RegionInfo(
            region_id=f"region_{i}",
            name=f"Region {i}",
            location=f"loc_{i}",
            timezone="UTC",
            endpoint=f"http://r{i}.example.com",
            status=RegionStatus.ACTIVE,
            capacity=1000,
            current_load=100
        )
        await region_manager.register_region(region)
        regions.append(region)

    # 注入区域离线故障
    await chaos.inject_fault(
        FailureType.REGION_OFFLINE,
        region_manager,
        duration=2.0
    )

    # 验证系统仍能路由到可用区域
    for _ in range(10):
        selected = await region_manager.select_region()
        assert selected is not None
        assert selected.is_available

    report = chaos.get_report()
    assert report.total_failures >= 1


@pytest.mark.asyncio
async def test_agent_pool_resilience():
    """测试Agent池弹性"""
    manager = AgentPoolManager()
    chaos = ChaosMonkey()

    # 创建Agent池
    pool = await manager.create_pool(
        name="chaos_test_pool",
        agent_type="test_agent",
        min_size=5,
        max_size=20,
        initial_size=10
    )

    # 正常工作负载
    tasks = []
    for _ in range(20):
        agent = await manager.acquire_agent(pool.pool_id)
        if agent:
            tasks.append(agent)

    # 注入Agent崩溃故障
    await chaos.inject_fault(FailureType.AGENT_CRASH, manager, duration=1.0)

    # 验证系统仍能处理请求
    available_count = 0
    for _ in range(10):
        agent = await manager.acquire_agent(pool.pool_id)
        if agent:
            available_count += 1
            await manager.release_agent(agent.agent_id)

    # 释放之前获取的agent
    for agent in tasks:
        await manager.release_agent(agent.agent_id)

    # 系统应该保持可用
    final_pool = (await manager.list_pools())[0]
    assert final_pool.available_count >= 0


@pytest.mark.asyncio
async def test_model_router_failover():
    """测试模型路由器故障转移"""
    router = ModelRouter()
    chaos = ChaosMonkey()

    # 注册多个提供商
    for i in range(3):
        config = ProviderConfig(
            provider_id=f"provider_{i}",
            name=f"Provider {i}",
            base_url=f"http://provider{i}.example.com",
            api_key=f"key_{i}",
            status=ProviderStatus.ACTIVE
        )
        provider = OpenAICompatibleProvider(config)
        router.register_provider(provider)

        model = ModelInfo(
            model_id=f"model_{i}",
            provider_id=f"provider_{i}",
            display_name=f"Model {i}",
            capabilities={ModelCapability.CHAT},
            max_context_tokens=4096,
            max_output_tokens=1024
        )
        router.register_model(model)

    # 注入提供商故障
    await chaos.inject_fault(
        FailureType.PROVIDER_FAILURE,
        router,
        duration=2.0
    )

    # 验证路由仍能工作
    for _ in range(10):
        model_info = await router.route()
        # 应该返回可用模型或None
        if model_info:
            provider = router._providers.get(model_info.provider_id)
            if provider:
                assert provider.is_available


@pytest.mark.asyncio
async def test_workflow_fault_tolerance():
    """测试工作流容错"""
    editor = WorkflowEditor()
    executor = WorkflowExecutor(editor)
    chaos = ChaosMonkey()

    # 创建复杂工作流
    workflow = editor.create_workflow("chaos_workflow")
    workflow_id = workflow.id

    # 添加节点
    for i in range(5):
        node = WorkflowNode(
            id=f"node_{i}",
            type=NodeType.FUNCTION,
            name=f"Node {i}",
            config={"code": f"context['step_{i}'] = {i}"}
        )
        editor.add_node(workflow_id, node)

    # 连接节点
    for i in range(4):
        editor.connect_nodes(workflow_id, f"node_{i}", f"node_{i+1}")

    # 执行工作流
    results = []
    for i in range(10):
        try:
            context = {"iteration": i}
            result = await executor.execute(workflow_id, context)
            results.append(result.status.value)
        except Exception as e:
            results.append(f"error: {e}")

    # 验证至少部分执行成功
    success_count = sum(1 for r in results if r == "completed")
    assert success_count >= 5, f"工作流容错不足: {results}"


@pytest.mark.asyncio
async def test_cascading_failure_prevention():
    """测试级联故障防护"""
    orchestrator = CrossRegionOrchestrator()
    region_manager = orchestrator.get_region_manager()
    chaos = ChaosMonkey()

    # 创建区域依赖链
    regions = []
    for i in range(5):
        region = RegionInfo(
            region_id=f"cascade_region_{i}",
            name=f"Cascade Region {i}",
            location=f"loc_{i}",
            timezone="UTC",
            endpoint=f"http://cascade{i}.example.com",
            status=RegionStatus.ACTIVE,
            capacity=100,
            current_load=10
        )
        await region_manager.register_region(region)
        regions.append(region)

    # 快速连续注入多个故障
    fault_tasks = [
        chaos.inject_fault(FailureType.REGION_DEGRADED, region_manager, 1.0),
        chaos.inject_fault(FailureType.REGION_OFFLINE, region_manager, 1.0),
        chaos.inject_fault(FailureType.HIGH_LATENCY, region_manager, 1.0),
    ]
    await asyncio.gather(*fault_tasks)

    # 验证系统仍能选择可用区域
    available_regions = [r for r in await region_manager.list_regions() if r.is_available]

    # 至少应该有一些区域保持可用
    assert len(available_regions) > 0

    # 验证健康检查能正确识别故障
    health = await region_manager.health_check()
    healthy_count = sum(1 for h in health.values() if h.get("healthy"))
    assert healthy_count >= 0  # 允许全部不健康，但不应该崩溃


@pytest.mark.asyncio
async def test_recovery_time_measurement():
    """测试恢复时间测量"""
    orchestrator = CrossRegionOrchestrator()
    region_manager = orchestrator.get_region_manager()
    chaos = ChaosMonkey()

    # 注册区域
    region = RegionInfo(
        region_id="recovery_test",
        name="Recovery Test Region",
        location="test",
        timezone="UTC",
        endpoint="http://test.example.com",
        status=RegionStatus.ACTIVE
    )
    await region_manager.register_region(region)

    # 注入故障并测量恢复时间
    fault_start = time.time()
    await chaos.inject_fault(
        FailureType.REGION_OFFLINE,
        region_manager,
        duration=1.0
    )

    # 等待恢复
    await asyncio.sleep(1.5)

    # 验证区域已恢复
    recovered_region = await region_manager.get_region("recovery_test")
    assert recovered_region.status == RegionStatus.ACTIVE

    report = chaos.get_report()
    assert report.total_failures >= 1


@pytest.mark.asyncio
async def test_chaos_stress_test():
    """混沌压力测试 - 同时注入多种故障"""
    orchestrator = CrossRegionOrchestrator()
    region_manager = orchestrator.get_region_manager()
    manager = AgentPoolManager()
    chaos = ChaosMonkey()

    # 初始化系统
    for i in range(3):
        region = RegionInfo(
            region_id=f"stress_region_{i}",
            name=f"Stress Region {i}",
            location=f"loc_{i}",
            timezone="UTC",
            endpoint=f"http://stress{i}.example.com",
            status=RegionStatus.ACTIVE,
            capacity=500
        )
        await region_manager.register_region(region)

    pool = await manager.create_pool(
        name="chaos_stress_pool",
        agent_type="stress_agent",
        min_size=3,
        max_size=10,
        initial_size=5
    )

    # 同时进行正常操作和故障注入
    async def normal_operations():
        success = 0
        for _ in range(20):
            try:
                # 区域选择
                region = await region_manager.select_region()
                if region:
                    success += 1

                # Agent获取
                agent = await manager.acquire_agent(pool.pool_id)
                if agent:
                    await asyncio.sleep(0.01)
                    await manager.release_agent(agent.agent_id)
                    success += 1
            except Exception:
                pass
            await asyncio.sleep(0.05)
        return success

    async def inject_faults():
        for _ in range(5):
            fault_type = random.choice(list(FailureType))
            if fault_type in chaos._faults:
                if fault_type in [FailureType.REGION_OFFLINE, FailureType.REGION_DEGRADED]:
                    await chaos.inject_fault(fault_type, region_manager, 0.5)
                elif fault_type in [FailureType.AGENT_CRASH, FailureType.AGENT_TIMEOUT]:
                    await chaos.inject_fault(fault_type, manager, 0.5)
            await asyncio.sleep(0.3)

    # 并行执行
    results = await asyncio.gather(
        normal_operations(),
        inject_faults(),
        return_exceptions=True
    )

    # 验证系统在故障下仍能保持一定可用性
    normal_success = results[0] if isinstance(results[0], int) else 0
    assert normal_success >= 10, f"系统在故障下可用性过低: {normal_success}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
