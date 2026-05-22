"""
Hermes Cluster v6.0 - 第八轮全覆盖测试
目标：极端边界、并发压力、混沌工程、业务闭环

测试维度：
1. 数值边界测试 - 极端参数、溢出、负数、零值
2. 并发竞态测试 - 高并发操作、死锁检测、数据一致性
3. 故障注入测试 - 随机故障、级联失败、自愈验证
4. 业务闭环测试 - 完整业务流程、状态机全覆盖
5. 资源泄漏测试 - 内存、连接、信号量泄漏
"""

import asyncio
import pytest
import time
import random
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# 导入被测模块
import sys
sys.path.insert(0, '/workspace/hermes-cluster/src')

from hermes_cluster.scale.pool import (
    AgentPoolManager, AgentSpec, AgentState, ScalingAction
)
from hermes_cluster.lowcode.editor import (
    WorkflowEditor, WorkflowExecutor, WorkflowValidator,
    VisualNode, VisualWorkflow, NodeType, NodeStatus, Connection
)
from hermes_cluster.geo.region import (
    RegionManager, RegionInfo, RegionStatus, DataSyncEngine,
    SyncMode, RoutingStrategy, FailoverManager, GeoRoutingService,
    CrossRegionOrchestrator
)
from hermes_cluster.models.router import (
    ModelRouter, ModelProvider, ModelInfo, ProviderConfig,
    ProviderStatus, ChatRequest, ChatMessage, ChatResponse,
    RouteStrategy, ModelCapability
)


# ==================== 第一部分：数值边界测试 ====================

class TestNumericBoundaries:
    """数值边界测试：极端参数、溢出、负数、零值"""

    @pytest.mark.asyncio
    async def test_pool_size_boundaries(self):
        """Agent池大小边界：0, 负数, 极大值, 溢出"""
        manager = AgentPoolManager()

        # 边界1: min_size=0
        pool1 = await manager.create_pool("pool_zero", "test", min_size=0, max_size=10, initial_size=0)
        assert pool1.current_size == 0, "BUG: min_size=0, initial_size=0 应创建空池"

        # 边界2: max_size < min_size (应自动修正)
        pool2 = await manager.create_pool("pool_invalid", "test", min_size=10, max_size=2, initial_size=5)
        # 期望：min_size被修正为<=max_size，或抛出异常
        assert pool2.min_size <= pool2.max_size, "BUG: min_size不应大于max_size"

        # 边界3: initial_size负数
        pool3 = await manager.create_pool("pool_negative", "test", min_size=2, max_size=10, initial_size=-5)
        assert pool3.current_size >= 0, "BUG: current_size不应为负数"

        # 边界4: 极大值 (不应OOM)
        pool4 = await manager.create_pool("pool_huge", "test", min_size=1, max_size=1000000, initial_size=1)
        assert pool4.max_size == 1000000, "BUG: 应接受大max_size"

    @pytest.mark.asyncio
    async def test_agent_task_count_boundaries(self):
        """Agent任务计数边界：0, max_tasks边界, 超限"""
        manager = AgentPoolManager()
        # 注意：create_pool签名是 (name, agent_type, min_size, max_size, initial_size)
        # agent_type是字符串，max_tasks在AgentSpec中默认为10
        pool = await manager.create_pool("pool1", "test_agent", min_size=1, max_size=5, initial_size=1)

        # 获取agent
        agent = await manager.acquire_agent(pool.pool_id)
        assert agent is not None

        # 边界1: current_tasks应从0开始
        assert agent.current_tasks >= 0, "BUG: current_tasks不应为负数"

        # 边界2: 连续acquire直到达到max_tasks (默认10)
        for _ in range(agent.spec.max_tasks - 1):
            await manager.acquire_agent(pool.pool_id)

        # 边界3: 超过max_tasks后应返回None或创建新agent
        agent_over = await manager.acquire_agent(pool.pool_id)
        # 期望：要么返回None，要么创建新agent
        # 如果返回同一个agent，检查current_tasks是否超过max_tasks
        if agent_over and agent_over.agent_id == agent.agent_id:
            assert agent_over.current_tasks <= agent_over.spec.max_tasks, \
                f"BUG: current_tasks={agent_over.current_tasks} 超过 max_tasks={agent_over.spec.max_tasks}"

    @pytest.mark.asyncio
    async def test_workflow_timeout_boundaries(self):
        """工作流超时边界：0, 负数, 极小值, 极大值"""
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)

        wf = editor.create_workflow("timeout_test")
        node = VisualNode(id="n1", type=NodeType.FUNCTION, name="Test",
                         config={"code": "result = 1"})
        editor.add_node(wf.id, node)

        # 边界1: timeout=0 (应立即超时或使用默认值)
        result1 = await executor.execute(wf.id, {}, timeout=0)
        # 期望：要么立即超时，要么使用默认值成功执行
        assert result1.status in [NodeStatus.COMPLETED, NodeStatus.TIMEOUT], \
            f"BUG: timeout=0导致异常状态 {result1.status}"

        # 边界2: timeout负数 (应使用默认值)
        result2 = await executor.execute(wf.id, {}, timeout=-1)
        assert result2.status == NodeStatus.COMPLETED, \
            f"BUG: timeout负数应使用默认值，实际状态 {result2.status}"

        # 边界3: timeout极大值
        result3 = await executor.execute(wf.id, {}, timeout=999999)
        assert result3.status == NodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_region_capacity_boundaries(self):
        """区域容量边界：0, 负数, 超限"""
        manager = RegionManager()

        # 边界1: capacity=0
        region1 = RegionInfo(
            region_id="r1", name="R1", location="loc1", timezone="UTC",
            endpoint="http://r1", capacity=0, current_load=0
        )
        await manager.register_region(region1)
        # load_ratio应处理capacity=0的情况
        assert region1.load_ratio == 0.0, "BUG: capacity=0时load_ratio应为0"

        # 边界2: current_load > capacity
        region2 = RegionInfo(
            region_id="r2", name="R2", location="loc2", timezone="UTC",
            endpoint="http://r2", capacity=10, current_load=20
        )
        await manager.register_region(region2)
        # load_ratio可以>1，表示超载
        assert region2.load_ratio == 2.0, "BUG: load_ratio应允许>1表示超载"

        # 边界3: current_load负数
        region3 = RegionInfo(
            region_id="r3", name="R3", location="loc3", timezone="UTC",
            endpoint="http://r3", capacity=10, current_load=-5
        )
        await manager.register_region(region3)
        # 应自动修正或允许负数
        # load_ratio会是负数，这可能是个问题
        assert region3.load_ratio >= 0, "BUG: current_load负数导致load_ratio为负"


# ==================== 第二部分：并发竞态测试 ====================

class TestConcurrencyRaceConditions:
    """并发竞态测试：高并发操作、死锁检测、数据一致性"""

    @pytest.mark.asyncio
    async def test_pool_concurrent_acquire_release(self):
        """并发acquire/release竞态条件"""
        manager = AgentPoolManager()
        spec = AgentSpec(agent_type="test", max_tasks=100)
        pool = await manager.create_pool("concurrent_pool", spec, min_size=5, max_size=20, initial_size=5)

        acquired_agents = []
        errors = []

        async def acquire_task(task_id: int):
            try:
                for _ in range(10):
                    agent = await manager.acquire_agent(pool.pool_id)
                    if agent:
                        acquired_agents.append(agent.agent_id)
                        await asyncio.sleep(0.001)
                        await manager.release_agent(agent.agent_id)
            except Exception as e:
                errors.append(f"Task {task_id}: {e}")

        # 100个并发任务
        tasks = [acquire_task(i) for i in range(100)]
        await asyncio.gather(*tasks)

        assert len(errors) == 0, f"BUG: 并发错误 - {errors[:5]}"

        # 验证最终状态一致性
        final_pool = manager._pools.get(pool.pool_id)
        total_tasks = sum(a.current_tasks for a in final_pool.agents.values())
        assert total_tasks == 0, f"BUG: 并发后current_tasks不一致，总任务数={total_tasks}"

    @pytest.mark.asyncio
    async def test_pool_concurrent_scale(self):
        """并发scale_pool竞态条件"""
        manager = AgentPoolManager()
        spec = AgentSpec(agent_type="test", max_tasks=10)
        pool = await manager.create_pool("scale_pool", spec, min_size=2, max_size=50, initial_size=5)

        errors = []

        async def scale_task(target: int):
            try:
                await manager.scale_pool(pool.pool_id, target)
            except Exception as e:
                errors.append(f"Scale to {target}: {e}")

        # 并发执行冲突的缩放操作
        tasks = [scale_task(random.randint(0, 30)) for _ in range(50)]
        await asyncio.gather(*tasks)

        assert len(errors) == 0, f"BUG: 并发缩放错误 - {errors[:5]}"

        # 验证最终状态在合法范围内
        final_pool = manager._pools.get(pool.pool_id)
        assert final_pool.current_size >= 0, f"BUG: current_size为负数 {final_pool.current_size}"
        assert final_pool.current_size <= final_pool.max_size, \
            f"BUG: current_size超过max_size {final_pool.current_size} > {final_pool.max_size}"

    @pytest.mark.asyncio
    async def test_region_concurrent_register_unregister(self):
        """并发注册/注销区域竞态条件"""
        manager = RegionManager()
        errors = []

        async def register_task(i: int):
            try:
                region = RegionInfo(
                    region_id=f"region_{i}", name=f"Region {i}",
                    location="loc", timezone="UTC", endpoint=f"http://r{i}"
                )
                await manager.register_region(region)
                await asyncio.sleep(0.001)
                await manager.unregister_region(f"region_{i}")
            except Exception as e:
                errors.append(f"Task {i}: {e}")

        tasks = [register_task(i) for i in range(50)]
        await asyncio.gather(*tasks)

        assert len(errors) == 0, f"BUG: 并发注册/注销错误 - {errors[:5]}"

        # 验证最终状态
        regions = await manager.list_regions()
        assert len(regions) == 0, f"BUG: 并发后区域残留 {len(regions)} 个"

    @pytest.mark.asyncio
    async def test_workflow_concurrent_execution(self):
        """并发工作流执行竞态条件"""
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)

        # 创建多个工作流
        workflow_ids = []
        for i in range(10):
            wf = editor.create_workflow(f"concurrent_wf_{i}")
            for j in range(5):
                node = VisualNode(id=f"n_{i}_{j}", type=NodeType.FUNCTION,
                                name=f"Node {j}", config={"code": f"result = {i}+{j}"})
                editor.add_node(wf.id, node)
            workflow_ids.append(wf.id)

        errors = []

        async def execute_task(wf_id: str):
            try:
                result = await executor.execute(wf_id, {})
                if result.status != NodeStatus.COMPLETED:
                    errors.append(f"Workflow {wf_id} failed: {result.error}")
            except Exception as e:
                errors.append(f"Workflow {wf_id} exception: {e}")

        tasks = [execute_task(wf_id) for wf_id in workflow_ids]
        await asyncio.gather(*tasks)

        assert len(errors) == 0, f"BUG: 并发工作流执行错误 - {errors[:5]}"


# ==================== 第三部分：故障注入测试 ====================

class TestFaultInjection:
    """故障注入测试：随机故障、级联失败、自愈验证"""

    @pytest.mark.asyncio
    async def test_pool_agent_creation_failure(self):
        """Agent创建失败处理"""
        manager = AgentPoolManager()
        spec = AgentSpec(agent_type="test", max_tasks=10)

        # 正常创建
        pool = await manager.create_pool("fail_pool", spec, min_size=1, max_size=10, initial_size=1)

        # 模拟创建失败：通过注入异常
        original_create = manager._create_agent

        async def failing_create(p):
            raise RuntimeError("模拟Agent创建失败")

        manager._create_agent = failing_create

        # 尝试扩容，应优雅处理失败
        try:
            result = await manager.scale_pool(pool.pool_id, 5)
            # 如果没有抛出异常，检查结果
            if result:
                assert result.new_size >= pool.current_size, "BUG: 扩容失败后size不应减少"
        except RuntimeError:
            pass  # 预期行为：抛出异常
        finally:
            manager._create_agent = original_create

        # 验证池仍然可用
        agent = await manager.acquire_agent(pool.pool_id)
        assert agent is not None, "BUG: 创建失败后池应仍可用"

    @pytest.mark.asyncio
    async def test_region_failover(self):
        """区域故障转移"""
        orchestrator = CrossRegionOrchestrator()
        manager = orchestrator.get_region_manager()

        # 注册多个区域
        for i in range(3):
            region = RegionInfo(
                region_id=f"region_{i}", name=f"Region {i}",
                location=f"loc_{i}", timezone="UTC", endpoint=f"http://r{i}",
                priority=i
            )
            await manager.register_region(region)

        # 选择区域
        selected = await manager.select_region()
        assert selected is not None

        # 模拟主区域故障
        await manager.update_region_status(selected.region_id, RegionStatus.OFFLINE)

        # 再次选择，应自动切换到其他区域
        new_selected = await manager.select_region()
        assert new_selected is not None
        assert new_selected.region_id != selected.region_id, "BUG: 故障区域未被排除"

    @pytest.mark.asyncio
    async def test_model_router_failover(self):
        """模型路由故障转移"""
        router = ModelRouter()

        # 创建模拟Provider
        class MockProvider(ModelProvider):
            def __init__(self, config, should_fail=False):
                super().__init__(config)
                self.should_fail = should_fail

            async def chat_completion(self, request):
                if self.should_fail:
                    raise RuntimeError("模拟Provider失败")
                return ChatResponse(
                    request_id=request.request_id,
                    model=request.model,
                    provider_id=self.provider_id,
                    content="OK"
                )

            async def chat_completion_stream(self, request):
                if self.should_fail:
                    raise RuntimeError("模拟Provider失败")
                yield "OK"

        # 注册主Provider（会失败）
        fail_config = ProviderConfig(
            provider_id="fail_provider", name="Fail Provider",
            base_url="http://fail", api_key="key"
        )
        fail_provider = MockProvider(fail_config, should_fail=True)
        router.register_provider(fail_provider)

        # 注册备用Provider（正常）
        ok_config = ProviderConfig(
            provider_id="ok_provider", name="OK Provider",
            base_url="http://ok", api_key="key"
        )
        ok_provider = MockProvider(ok_config, should_fail=False)
        router.register_provider(ok_provider)

        # 注册模型
        router.register_model(ModelInfo(
            model_id="fail_model", provider_id="fail_provider",
            display_name="Fail Model", priority=100
        ))
        router.register_model(ModelInfo(
            model_id="ok_model", provider_id="ok_provider",
            display_name="OK Model", priority=50
        ))

        # 发送请求，应故障转移到ok_provider
        request = ChatRequest(model="fail_model", messages=[ChatMessage(role="user", content="test")])
        response = await router.chat(request)

        assert response.provider_id == "ok_provider", \
            f"BUG: 故障转移失败，实际provider={response.provider_id}"

    @pytest.mark.asyncio
    async def test_workflow_node_failure_propagation(self):
        """工作流节点失败传播"""
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)

        wf = editor.create_workflow("fail_propagation")

        # 创建链式节点：n1 -> n2 -> n3
        n1 = VisualNode(id="n1", type=NodeType.FUNCTION, name="N1",
                       config={"code": "result = 1"})
        n2 = VisualNode(id="n2", type=NodeType.FUNCTION, name="N2",
                       config={"code": "raise ValueError('N2失败')"})
        n3 = VisualNode(id="n3", type=NodeType.FUNCTION, name="N3",
                       config={"code": "result = 3"})

        editor.add_node(wf.id, n1)
        editor.add_node(wf.id, n2)
        editor.add_node(wf.id, n3)

        editor.connect_nodes(wf.id, "n1", "n2")
        editor.connect_nodes(wf.id, "n2", "n3")

        result = await executor.execute(wf.id, {})

        # 验证失败传播
        assert result.status == NodeStatus.FAILED
        assert "n1" in result.node_results
        assert result.node_results["n1"].status == NodeStatus.COMPLETED
        assert result.node_results["n2"].status == NodeStatus.FAILED
        # n3不应执行
        assert "n3" not in result.node_results or \
               result.node_results["n3"].status == NodeStatus.PENDING


# ==================== 第四部分：业务闭环测试 ====================

class TestBusinessFlowClosure:
    """业务闭环测试：完整业务流程、状态机全覆盖"""

    @pytest.mark.asyncio
    async def test_agent_full_lifecycle(self):
        """Agent完整生命周期：创建->忙碌->空闲->释放->销毁"""
        manager = AgentPoolManager()
        spec = AgentSpec(agent_type="lifecycle_test", max_tasks=5)
        pool = await manager.create_pool("lifecycle_pool", spec, min_size=1, max_size=5, initial_size=1)

        # 1. 初始状态：IDLE
        agents = list(pool.agents.values())
        assert len(agents) == 1
        assert agents[0].state == AgentState.IDLE

        # 2. Acquire -> BUSY
        agent = await manager.acquire_agent(pool.pool_id)
        assert agent.state == AgentState.BUSY

        # 3. 多次acquire直到max_tasks
        for _ in range(spec.max_tasks - 1):
            await manager.acquire_agent(pool.pool_id)
        assert agent.current_tasks == spec.max_tasks

        # 4. Release -> 逐渐减少任务
        for _ in range(spec.max_tasks):
            await manager.release_agent(agent.agent_id)
        assert agent.current_tasks == 0
        assert agent.state == AgentState.IDLE

        # 5. Scale down -> 销毁
        await manager.scale_pool(pool.pool_id, 0)
        assert pool.current_size == 0

    @pytest.mark.asyncio
    async def test_workflow_full_execution_flow(self):
        """工作流完整执行流程：创建->编辑->验证->执行->清理"""
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)

        # 1. 创建工作流
        wf = editor.create_workflow("full_flow_test")
        assert wf is not None

        # 2. 添加各种类型节点
        nodes = [
            VisualNode(id="start", type=NodeType.TRIGGER, name="Start"),
            VisualNode(id="task1", type=NodeType.TASK, name="Task1"),
            VisualNode(id="condition", type=NodeType.CONDITION, name="Check",
                      config={"condition": "True"}),
            VisualNode(id="branch_true", type=NodeType.FUNCTION, name="TrueBranch",
                      config={"code": "result = 'true'"}),
            VisualNode(id="branch_false", type=NodeType.FUNCTION, name="FalseBranch",
                      config={"code": "result = 'false'"}),
            VisualNode(id="parallel", type=NodeType.PARALLEL, name="Parallel"),
            VisualNode(id="end", type=NodeType.OUTPUT, name="End"),
        ]

        for node in nodes:
            editor.add_node(wf.id, node)

        # 3. 连接节点
        editor.connect_nodes(wf.id, "start", "task1")
        editor.connect_nodes(wf.id, "task1", "condition")
        editor.connect_nodes(wf.id, "condition", "branch_true", source_port="true")
        editor.connect_nodes(wf.id, "condition", "branch_false", source_port="false")
        editor.connect_nodes(wf.id, "branch_true", "end")

        # 4. 验证
        errors = await editor.validate_workflow(wf.id)
        # 可能有孤立节点警告，但不应有致命错误

        # 5. 执行
        result = await executor.execute(wf.id, {"input": "test"})
        assert result.status == NodeStatus.COMPLETED, f"BUG: 执行失败 - {result.error}"

        # 6. 清理
        editor.remove_node(wf.id, "start")
        assert "start" not in editor.get_workflow(wf.id).nodes

    @pytest.mark.asyncio
    async def test_region_full_lifecycle(self):
        """区域完整生命周期：注册->活跃->降级->故障转移->恢复->注销"""
        orchestrator = CrossRegionOrchestrator()
        manager = orchestrator.get_region_manager()

        # 1. 注册区域
        region = RegionInfo(
            region_id="lifecycle_region", name="Lifecycle Region",
            location="loc", timezone="UTC", endpoint="http://r1",
            priority=10
        )
        result = await manager.register_region(region)
        assert result.region_id == "lifecycle_region"

        # 2. 活跃状态
        await manager.update_region_status("lifecycle_region", RegionStatus.ACTIVE, load=50)
        r = await manager.get_region("lifecycle_region")
        assert r.status == RegionStatus.ACTIVE
        assert r.current_load == 50

        # 3. 降级状态
        await manager.update_region_status("lifecycle_region", RegionStatus.DEGRADED)
        r = await manager.get_region("lifecycle_region")
        assert r.status == RegionStatus.DEGRADED
        assert r.is_available  # DEGRADED仍可用

        # 4. 故障转移状态
        await manager.update_region_status("lifecycle_region", RegionStatus.FAILOVER)
        r = await manager.get_region("lifecycle_region")
        assert not r.is_available  # FAILOVER不可用

        # 5. 恢复
        await manager.update_region_status("lifecycle_region", RegionStatus.ACTIVE)
        r = await manager.get_region("lifecycle_region")
        assert r.is_available

        # 6. 注销
        await manager.unregister_region("lifecycle_region")
        r = await manager.get_region("lifecycle_region")
        assert r is None


# ==================== 第五部分：资源泄漏测试 ====================

class TestResourceLeaks:
    """资源泄漏测试：内存、连接、信号量泄漏"""

    @pytest.mark.asyncio
    async def test_pool_semaphore_leak(self):
        """Agent池信号量泄漏检测"""
        manager = AgentPoolManager()
        spec = AgentSpec(agent_type="semaphore_test", max_tasks=10)
        pool = await manager.create_pool("sem_pool", spec, min_size=1, max_size=5, initial_size=1)

        # 大量acquire/release操作
        for _ in range(100):
            agent = await manager.acquire_agent(pool.pool_id)
            if agent:
                await manager.release_agent(agent.agent_id)

        # 验证信号量状态
        # 如果有泄漏，后续acquire会阻塞
        acquired = []
        for _ in range(5):
            agent = await manager.acquire_agent(pool.pool_id)
            if agent:
                acquired.append(agent)

        # 应该能成功acquire
        assert len(acquired) > 0, "BUG: 信号量泄漏导致无法acquire"

        # 清理
        for agent in acquired:
            await manager.release_agent(agent.agent_id)

    @pytest.mark.asyncio
    async def test_execution_result_leak(self):
        """执行结果内存泄漏检测"""
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)

        wf = editor.create_workflow("leak_test")
        node = VisualNode(id="n1", type=NodeType.FUNCTION, name="Test",
                         config={"code": "result = 1"})
        editor.add_node(wf.id, node)

        # 大量执行
        for _ in range(600):
            await executor.execute(wf.id, {})

        # 验证结果缓存有限
        assert len(executor._execution_results) <= 500, \
            f"BUG: 执行结果泄漏，缓存数量={len(executor._execution_results)}"

    @pytest.mark.asyncio
    async def test_sync_task_leak(self):
        """同步任务内存泄漏检测"""
        orchestrator = CrossRegionOrchestrator()
        manager = orchestrator.get_region_manager()
        engine = DataSyncEngine(manager)

        # 注册区域
        region = RegionInfo(
            region_id="sync_region", name="Sync Region",
            location="loc", timezone="UTC", endpoint="http://r1"
        )
        await manager.register_region(region)

        # 创建分区
        await engine.create_partition("p1", "sync_region", [])

        # 大量同步任务
        for i in range(1100):
            await engine.sync_data("p1", {"data": i}, "sync_region")

        # 验证任务缓存有限
        assert len(engine._sync_tasks) <= 1000, \
            f"BUG: 同步任务泄漏，缓存数量={len(engine._sync_tasks)}"


# ==================== 第六部分：特殊场景测试 ====================

class TestSpecialScenarios:
    """特殊场景测试：极端组合、边界交叉、异常恢复"""

    @pytest.mark.asyncio
    async def test_empty_pool_acquire(self):
        """空池acquire行为"""
        manager = AgentPoolManager()
        spec = AgentSpec(agent_type="empty_test", max_tasks=10)
        pool = await manager.create_pool("empty_pool", spec, min_size=0, max_size=5, initial_size=0)

        # 空池acquire应返回None或自动创建agent
        agent = await manager.acquire_agent(pool.pool_id)
        # 期望：返回None（因为min_size=0且没有agent）
        assert agent is None or pool.current_size > 0, \
            "BUG: 空池acquire行为不符合预期"

    @pytest.mark.asyncio
    async def test_circular_workflow_detection(self):
        """循环工作流检测"""
        editor = WorkflowEditor()

        wf = editor.create_workflow("circular")
        n1 = VisualNode(id="n1", type=NodeType.FUNCTION, name="N1", config={"code": "result=1"})
        n2 = VisualNode(id="n2", type=NodeType.FUNCTION, name="N2", config={"code": "result=2"})
        n3 = VisualNode(id="n3", type=NodeType.FUNCTION, name="N3", config={"code": "result=3"})

        editor.add_node(wf.id, n1)
        editor.add_node(wf.id, n2)
        editor.add_node(wf.id, n3)

        # 创建循环：n1 -> n2 -> n3 -> n1
        editor.connect_nodes(wf.id, "n1", "n2")
        editor.connect_nodes(wf.id, "n2", "n3")
        editor.connect_nodes(wf.id, "n3", "n1")

        # 验证应检测到循环
        is_valid, errors = WorkflowValidator.validate(wf)
        assert not is_valid, "BUG: 循环工作流验证通过"
        assert any("循环" in e for e in errors), f"BUG: 未检测到循环，错误: {errors}"

    @pytest.mark.asyncio
    async def test_deeply_nested_workflow(self):
        """深度嵌套工作流"""
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)

        wf = editor.create_workflow("deep_nested")

        # 创建100个节点的链
        prev_id = None
        for i in range(100):
            node = VisualNode(id=f"n{i}", type=NodeType.FUNCTION,
                            name=f"N{i}", config={"code": f"result = {i}"})
            editor.add_node(wf.id, node)
            if prev_id:
                editor.connect_nodes(wf.id, prev_id, f"n{i}")
            prev_id = f"n{i}"

        # 执行
        result = await executor.execute(wf.id, {}, timeout=60)
        assert result.status == NodeStatus.COMPLETED, \
            f"BUG: 深度嵌套工作流执行失败 - {result.error}"

    @pytest.mark.asyncio
    async def test_rapid_create_destroy(self):
        """快速创建销毁"""
        manager = AgentPoolManager()

        async def create_destroy_cycle(i: int):
            spec = AgentSpec(agent_type=f"rapid_{i}", max_tasks=10)
            pool = await manager.create_pool(f"rapid_pool_{i}", spec, min_size=1, max_size=5, initial_size=1)
            await asyncio.sleep(0.001)
            # 通过scale到0来销毁
            await manager.scale_pool(pool.pool_id, 0)

        tasks = [create_destroy_cycle(i) for i in range(20)]
        await asyncio.gather(*tasks)

        # 验证无残留
        assert len(manager._pools) == 20  # pools仍存在，但size=0


# ==================== 运行入口 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
