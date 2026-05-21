"""
Hermes Cluster - 第五轮极端场景测试
更长时间、更大规模、更错综复杂、更高难度、更混乱的场景
目标：发现系统在极限压力下的深层缺陷
"""

import asyncio
import sys
import os
import logging
import time
import random
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from hermes_cluster.geo.region import (
    CrossRegionOrchestrator, RegionInfo, RegionStatus,
    DataSyncEngine, SyncMode
)
from hermes_cluster.scale.pool import (
    AgentPoolManager, AgentState, AutoScaler, ScalingPolicy
)
from hermes_cluster.lowcode.editor import (
    WorkflowEditor, WorkflowNode, NodeType, WorkflowExecutor,
    NodeStatus
)
from hermes_cluster.models.router import (
    ModelRouter, ProviderConfig, ProviderStatus, ModelInfo,
    ModelCapability, OpenAICompatibleProvider
)


# ============================================================
# 场景1: 并发风暴 - 极端并发下的系统稳定性
# ============================================================

class TestConcurrencyStorm:
    """并发风暴测试 - 模拟突发流量"""

    @pytest.mark.asyncio
    async def test_massive_concurrent_agent_operations(self):
        """1000个并发Agent操作"""
        manager = AgentPoolManager()
        pool = await manager.create_pool(
            name="storm_pool", agent_type="test",
            min_size=10, max_size=100, initial_size=20
        )

        async def agent_operation(i: int):
            """单个agent操作"""
            try:
                agent = await manager.acquire_agent(pool.pool_id)
                if agent:
                    await asyncio.sleep(random.uniform(0.001, 0.01))
                    await manager.release_agent(agent.agent_id)
                    return True
                return False
            except Exception as e:
                logging.error(f"Agent operation {i} failed: {e}")
                return False

        # 1000个并发操作
        tasks = [agent_operation(i) for i in range(1000)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        failure_count = sum(1 for r in results if r is False or isinstance(r, Exception))

        logging.info(f"Concurrent agent operations: {success_count} success, {failure_count} failure")

        # 验证pool状态一致性
        final_pool = (await manager.list_pools())[0]
        assert final_pool.current_size >= 0
        assert final_pool.current_size <= 100

    @pytest.mark.asyncio
    async def test_concurrent_region_operations(self):
        """并发区域操作风暴"""
        orchestrator = CrossRegionOrchestrator()
        manager = orchestrator.get_region_manager()

        async def region_operation(i: int):
            """随机区域操作"""
            try:
                op = random.choice(["register", "update", "select", "health"])
                if op == "register":
                    region = RegionInfo(
                        region_id=f"storm_{i}", name=f"Storm {i}",
                        location=f"loc_{i % 10}", timezone="UTC",
                        endpoint=f"http://storm{i}.com"
                    )
                    await manager.register_region(region)
                elif op == "update":
                    status = random.choice(list(RegionStatus))
                    await manager.update_region_status(f"storm_{i}", status)
                elif op == "select":
                    await manager.select_region()
                else:
                    await manager.health_check()
                return True
            except Exception as e:
                return False

        # 500个并发区域操作
        tasks = [region_operation(i) for i in range(500)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        logging.info(f"Concurrent region operations: {success_count}/500 success")

    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self):
        """并发工作流执行风暴"""
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)

        # 创建一个复杂工作流
        workflow = editor.create_workflow("storm_workflow")
        wf_id = workflow.id

        for i in range(10):
            node = WorkflowNode(
                id=f"n{i}", type=NodeType.FUNCTION, name=f"N{i}",
                config={"code": f"context['v{i}'] = {i}"}
            )
            editor.add_node(wf_id, node)

        async def execute_workflow(i: int):
            """执行工作流"""
            try:
                context = {}
                result = await executor.execute(wf_id, context)
                return result.status == NodeStatus.COMPLETED
            except Exception as e:
                return False

        # 200个并发工作流执行
        tasks = [execute_workflow(i) for i in range(200)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        logging.info(f"Concurrent workflow executions: {success_count}/200 success")


# ============================================================
# 场景2: 资源边界 - 极限资源压力测试
# ============================================================

class TestResourceBoundaries:
    """资源边界测试"""

    @pytest.mark.asyncio
    async def test_agent_pool_max_capacity(self):
        """Agent池最大容量边界"""
        manager = AgentPoolManager()
        pool = await manager.create_pool(
            name="max_pool", agent_type="test",
            min_size=1, max_size=1000, initial_size=1
        )

        # 逐步扩容到最大
        for target in [100, 500, 1000]:
            await manager.scale_pool(pool.pool_id, target)
            current_pool = (await manager.list_pools())[0]
            assert current_pool.current_size == target, f"Failed to scale to {target}"
            logging.info(f"Scaled to {target} agents")

        # 尝试超过最大值
        await manager.scale_pool(pool.pool_id, 2000)
        current_pool = (await manager.list_pools())[0]
        assert current_pool.current_size == 1000, "Should be capped at max_size"

    @pytest.mark.asyncio
    async def test_memory_pressure_with_many_workflows(self):
        """大量工作流内存压力测试"""
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)

        # 创建100个工作流
        workflow_ids = []
        for i in range(100):
            workflow = editor.create_workflow(f"mem_test_{i}")
            wf_id = workflow.id

            for j in range(5):
                node = WorkflowNode(
                    id=f"n{j}", type=NodeType.FUNCTION, name=f"N{j}",
                    config={"code": f"context['data_{j}'] = list(range(1000))"}
                )
                editor.add_node(wf_id, node)

            workflow_ids.append(wf_id)

        # 执行所有工作流
        async def run_workflow(wf_id: str):
            context = {}
            result = await executor.execute(wf_id, context)
            return result.status == NodeStatus.COMPLETED

        tasks = [run_workflow(wf_id) for wf_id in workflow_ids]
        results = await asyncio.gather(*tasks)

        success_count = sum(results)
        logging.info(f"Memory pressure test: {success_count}/100 success")

    @pytest.mark.asyncio
    async def test_execution_result_history_limit(self):
        """执行结果历史限制测试"""
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)

        workflow = editor.create_workflow("history_test")
        wf_id = workflow.id

        node = WorkflowNode(
            id="n1", type=NodeType.FUNCTION, name="N1",
            config={"code": "pass"}
        )
        editor.add_node(wf_id, node)

        # 执行600次，超过500的限制
        for i in range(600):
            result = await executor.execute(wf_id, {})
            assert result.status == NodeStatus.COMPLETED

        # 验证历史记录被限制
        assert len(executor._execution_results) <= 500
        logging.info(f"Execution history size: {len(executor._execution_results)}")


# ============================================================
# 场景3: 故障级联 - 多层级故障传播测试
# ============================================================

class TestCascadingFailures:
    """故障级联测试"""

    @pytest.mark.asyncio
    async def test_region_failure_cascade_to_pool(self):
        """区域故障级联到Agent池"""
        orchestrator = CrossRegionOrchestrator()
        region_manager = orchestrator.get_region_manager()
        pool_manager = AgentPoolManager()

        # 创建多个区域
        for i in range(5):
            region = RegionInfo(
                region_id=f"cascade_{i}", name=f"Cascade {i}",
                location=f"loc_{i}", timezone="UTC",
                endpoint=f"http://cascade{i}.com"
            )
            await region_manager.register_region(region)

        # 创建依赖这些区域的pools
        pools = []
        for i in range(5):
            pool = await pool_manager.create_pool(
                name=f"cascade_pool_{i}", agent_type="test",
                min_size=2, max_size=10, initial_size=5
            )
            pools.append(pool)

        # 级联故障：关闭区域，同时操作pools
        async def cascade_failure(i: int):
            await region_manager.update_region_status(f"cascade_{i}", RegionStatus.OFFLINE)
            # 尝试从对应pool获取agent
            agent = await pool_manager.acquire_agent(pools[i].pool_id)
            if agent:
                await pool_manager.release_agent(agent.agent_id)

        tasks = [cascade_failure(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # 验证系统仍可用
        available_regions = [r for r in await region_manager.list_regions() if r.is_available]
        logging.info(f"Available regions after cascade: {len(available_regions)}")

    @pytest.mark.asyncio
    async def test_workflow_failure_isolation(self):
        """工作流故障隔离测试"""
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)

        # 创建多个独立工作流
        workflows = []
        for i in range(10):
            workflow = editor.create_workflow(f"iso_{i}")
            wf_id = workflow.id

            # 奇数工作流包含错误节点
            if i % 2 == 1:
                node = WorkflowNode(
                    id="error", type=NodeType.FUNCTION, name="Error",
                    config={"code": "raise ValueError('intentional error')"}
                )
            else:
                node = WorkflowNode(
                    id="ok", type=NodeType.FUNCTION, name="OK",
                    config={"code": "context['ok'] = True"}
                )

            editor.add_node(wf_id, node)
            workflows.append(wf_id)

        # 并发执行所有工作流
        async def run_isolated(wf_id: str):
            context = {}
            result = await executor.execute(wf_id, context)
            return result.status

        results = await asyncio.gather(*[run_isolated(wf_id) for wf_id in workflows])

        # 验证：偶数工作流应成功，奇数应失败
        for i, status in enumerate(results):
            if i % 2 == 0:
                assert status == NodeStatus.COMPLETED, f"Workflow {i} should succeed"
            else:
                assert status == NodeStatus.FAILED, f"Workflow {i} should fail"

        logging.info("Failure isolation test passed")


# ============================================================
# 场景4: 长时间运行 - 稳定性测试
# ============================================================

class TestLongRunningStability:
    """长时间运行稳定性测试"""

    @pytest.mark.asyncio
    async def test_sustained_load(self):
        """持续负载测试（模拟长时间运行）"""
        manager = AgentPoolManager()
        pool = await manager.create_pool(
            name="sustained", agent_type="test",
            min_size=5, max_size=50, initial_size=10
        )

        operation_count = 0
        error_count = 0

        # 模拟10秒持续负载（压缩版长时间测试）
        start_time = time.time()
        while time.time() - start_time < 10:
            try:
                agent = await manager.acquire_agent(pool.pool_id)
                if agent:
                    await asyncio.sleep(0.001)
                    await manager.release_agent(agent.agent_id)
                    operation_count += 1
                else:
                    await asyncio.sleep(0.01)
            except Exception as e:
                error_count += 1

        logging.info(f"Sustained load: {operation_count} ops, {error_count} errors")
        assert error_count < operation_count * 0.01  # 错误率<1%

    @pytest.mark.asyncio
    async def test_repeated_scale_operations(self):
        """重复扩缩容操作稳定性"""
        manager = AgentPoolManager()
        pool = await manager.create_pool(
            name="scale_stress", agent_type="test",
            min_size=1, max_size=100, initial_size=10
        )

        # 100次随机扩缩容
        for i in range(100):
            target = random.randint(1, 100)
            await manager.scale_pool(pool.pool_id, target)

        final_pool = (await manager.list_pools())[0]
        assert final_pool.current_size >= 1
        assert final_pool.current_size <= 100
        logging.info(f"Final pool size after 100 scale ops: {final_pool.current_size}")


# ============================================================
# 场景5: 混乱注入 - 随机故障注入
# ============================================================

class TestChaosInjection:
    """混乱注入测试"""

    @pytest.mark.asyncio
    async def test_random_operation_mix(self):
        """随机操作混合测试"""
        orchestrator = CrossRegionOrchestrator()
        region_manager = orchestrator.get_region_manager()
        pool_manager = AgentPoolManager()
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)

        # 创建基础资源
        for i in range(3):
            region = RegionInfo(
                region_id=f"chaos_{i}", name=f"Chaos {i}",
                location=f"loc_{i}", timezone="UTC",
                endpoint=f"http://chaos{i}.com"
            )
            await region_manager.register_region(region)

        pool = await pool_manager.create_pool(
            name="chaos_pool", agent_type="test",
            min_size=5, max_size=30, initial_size=10
        )

        workflow = editor.create_workflow("chaos_wf")
        wf_id = workflow.id
        node = WorkflowNode(id="n1", type=NodeType.FUNCTION, name="N1", config={"code": "pass"})
        editor.add_node(wf_id, node)

        async def random_operation():
            """随机操作"""
            op = random.choice([
                "region_select", "region_update", "agent_acquire",
                "agent_release", "workflow_execute", "pool_scale"
            ])
            try:
                if op == "region_select":
                    await region_manager.select_region()
                elif op == "region_update":
                    status = random.choice(list(RegionStatus))
                    await region_manager.update_region_status(f"chaos_{random.randint(0,2)}", status)
                elif op == "agent_acquire":
                    await pool_manager.acquire_agent(pool.pool_id)
                elif op == "agent_release":
                    # 尝试释放随机agent（可能不存在）
                    await pool_manager.release_agent(f"random_{random.randint(0,100)}")
                elif op == "workflow_execute":
                    await executor.execute(wf_id, {})
                elif op == "pool_scale":
                    await pool_manager.scale_pool(pool.pool_id, random.randint(5, 30))
                return True
            except Exception:
                return False

        # 300个随机操作
        tasks = [random_operation() for _ in range(300)]
        results = await asyncio.gather(*tasks)

        success_count = sum(results)
        logging.info(f"Chaos injection: {success_count}/300 operations succeeded")

        # 系统应保持稳定
        final_pool = (await pool_manager.list_pools())[0]
        assert final_pool.current_size >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
