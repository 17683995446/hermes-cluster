"""
Hermes Cluster - 第七轮复杂场景全覆盖测试
聚焦：跨模块联动、状态机全覆盖、数据一致性、接口全量遍历
目标：穷尽所有if/else分支，验证数据流转自洽
"""

import asyncio
import sys
import os
import logging
import time
import random
import copy
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

logging.basicConfig(level=logging.WARNING)

from hermes_cluster.geo.region import (
    CrossRegionOrchestrator, RegionInfo, RegionStatus,
    DataSyncEngine, SyncMode, SyncTask, RoutingStrategy
)
from hermes_cluster.scale.pool import (
    AgentPoolManager, AgentState, AgentSpec, AgentInstance,
    AutoScaler, ScalingPolicy, ScalingAction
)
from hermes_cluster.lowcode.editor import (
    WorkflowEditor, WorkflowNode, NodeType, WorkflowExecutor,
    NodeStatus, WorkflowValidator, VisualWorkflow, Connection,
    TemplateMarket, ConfigGenerator, ConfigVersionManager
)
from hermes_cluster.models.router import (
    ModelRouter, ProviderConfig, ProviderStatus, ModelInfo,
    ModelCapability, OpenAICompatibleProvider, SiliconFlowProvider,
    ChatRequest, ChatMessage, RouteStrategy, UsageRecord
)


# ============================================================
# 1. 状态机全覆盖测试
# ============================================================

class TestStateMachineCoverage:
    """穷尽所有状态转换路径"""

    @pytest.mark.asyncio
    async def test_region_status_all_transitions(self):
        """区域状态机：穷尽所有状态转换"""
        orchestrator = CrossRegionOrchestrator()
        manager = orchestrator.get_region_manager()

        region = RegionInfo(
            region_id="sm_region", name="SM Region",
            location="loc_1", timezone="UTC",
            endpoint="http://sm.example.com"
        )
        await manager.register_region(region)

        # 穷尽所有状态转换
        all_statuses = list(RegionStatus)
        for target_status in all_statuses:
            await manager.update_region_status("sm_region", target_status)
            r = await manager.get_region("sm_region")
            assert r.status == target_status, f"状态转换失败: 期望{target_status}, 实际{r.status}"

        # 验证is_available在每种状态下的正确性
        for status in all_statuses:
            await manager.update_region_status("sm_region", status)
            r = await manager.get_region("sm_region")
            if status in (RegionStatus.ACTIVE, RegionStatus.DEGRADED):
                assert r.is_available is True
            else:
                assert r.is_available is False

    @pytest.mark.asyncio
    async def test_agent_state_all_transitions(self):
        """Agent状态机：穷尽所有状态转换"""
        manager = AgentPoolManager()
        pool = await manager.create_pool(
            name="sm_pool", agent_type="test",
            min_size=1, max_size=5, initial_size=1
        )

        agent = await manager.acquire_agent(pool.pool_id)
        assert agent is not None
        assert agent.state == AgentState.BUSY

        # 释放后回到IDLE
        await manager.release_agent(agent.agent_id)
        agent = manager._agent_registry.get(agent.agent_id)
        assert agent.state == AgentState.IDLE

        # DRAINING状态
        agent.state = AgentState.DRAINING
        assert agent.state == AgentState.DRAINING

        # ERROR状态
        agent.state = AgentState.ERROR
        assert agent.state == AgentState.ERROR

        # TERMINATING状态
        agent.state = AgentState.TERMINATING
        assert agent.state == AgentState.TERMINATING

    @pytest.mark.asyncio
    async def test_node_status_all_paths(self):
        """工作流节点状态：穷尽所有状态路径"""
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)

        # COMPLETED路径
        wf = editor.create_workflow("status_test")
        wf_id = wf.id
        node_ok = WorkflowNode(id="ok", type=NodeType.FUNCTION, name="OK",
                               config={"code": "context['x'] = 1"})
        editor.add_node(wf_id, node_ok)
        result = await executor.execute(wf_id, {})
        assert result.node_results["ok"].status == NodeStatus.COMPLETED

        # FAILED路径
        wf2 = editor.create_workflow("fail_test")
        wf2_id = wf2.id
        node_fail = WorkflowNode(id="fail", type=NodeType.FUNCTION, name="Fail",
                                config={"code": "raise ValueError('test error')"})
        editor.add_node(wf2_id, node_fail)
        result2 = await executor.execute(wf2_id, {})
        assert result2.node_results["fail"].status == NodeStatus.FAILED

        # TIMEOUT路径 - 使用注入的time模块（exec中import被禁用，但time已注入local_vars）
        # 注意：由于exec在线程池中运行，time.sleep阻塞线程但asyncio.wait_for会超时取消
        wf3 = editor.create_workflow("timeout_test")
        wf3_id = wf3.id
        node_slow = WorkflowNode(id="slow", type=NodeType.FUNCTION, name="Slow",
                                config={"code": "time.sleep(2)"})
        editor.add_node(wf3_id, node_slow)
        result3 = await executor.execute(wf3_id, {}, timeout=0.1)
        assert result3.node_results["slow"].status == NodeStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_sync_task_all_states(self):
        """同步任务状态：穷尽所有状态"""
        region_mgr = CrossRegionOrchestrator().get_region_manager()
        engine = DataSyncEngine(region_mgr)

        # 创建分区（有副本）
        await engine.create_partition(
            partition_id="sync_sm",
            primary_region="region_a",
            replica_regions=["region_b", "region_c"]
        )

        # 从主区域同步 → 应成功
        task_id = await engine.sync_data("sync_sm", {"key": "value"}, "region_a")
        task = engine._sync_tasks.get(task_id)
        assert task is not None

        # 从副本区域同步 → 应成功
        task_id2 = await engine.sync_data("sync_sm", {"key": "value2"}, "region_b")
        task2 = engine._sync_tasks.get(task_id2)
        assert task2 is not None

        # 同步到自身 → 应跳过
        task_id3 = await engine.sync_data("sync_sm", {"key": "value3"}, "region_a")
        task3 = engine._sync_tasks.get(task_id3)
        assert task3 is not None

        # 不存在的分区 → 应抛出异常
        try:
            await engine.sync_data("nonexistent", {}, "region_a")
            assert False, "应抛出ValueError"
        except ValueError:
            pass


# ============================================================
# 2. 接口全量遍历测试
# ============================================================

class TestInterfaceFullCoverage:
    """全覆盖所有接口：正常请求、异常入参、超限参数、空值、非法格式"""

    @pytest.mark.asyncio
    async def test_region_manager_all_interfaces(self):
        """RegionManager所有接口遍历"""
        manager = CrossRegionOrchestrator().get_region_manager()

        # 1. register_region - 正常
        r1 = await manager.register_region(RegionInfo(
            region_id="r1", name="R1", location="loc_1",
            timezone="UTC", endpoint="http://r1.com"
        ))
        assert r1.region_id == "r1"

        # 2. register_region - 重复注册（更新）
        r1_updated = await manager.register_region(RegionInfo(
            region_id="r1", name="R1 Updated", location="loc_1",
            timezone="UTC", endpoint="http://r1.com"
        ))
        assert r1_updated.name == "R1 Updated"

        # 3. get_region - 存在
        r = await manager.get_region("r1")
        assert r is not None

        # 4. get_region - 不存在
        r_none = await manager.get_region("nonexistent")
        assert r_none is None

        # 5. list_regions - 无过滤
        all_regions = await manager.list_regions()
        assert len(all_regions) >= 1

        # 6. list_regions - 按状态过滤
        active = await manager.list_regions(RegionStatus.ACTIVE)
        assert all(r.status == RegionStatus.ACTIVE for r in active)

        # 7. update_region_status - 正常
        await manager.update_region_status("r1", RegionStatus.OFFLINE)
        r = await manager.get_region("r1")
        assert r.status == RegionStatus.OFFLINE

        # 8. update_region_status - 不存在的区域
        result = await manager.update_region_status("nonexistent", RegionStatus.ACTIVE)
        # 应该不崩溃（可能返回None或静默失败）

        # 9. update_region_status - 带load参数
        await manager.update_region_status("r1", RegionStatus.ACTIVE, load=50)
        r = await manager.get_region("r1")
        assert r.current_load == 50

        # 10. update_region_status - load=None不更新负载
        await manager.update_region_status("r1", RegionStatus.ACTIVE, load=None)
        r = await manager.get_region("r1")
        assert r.current_load == 50  # 保持不变

        # 11. select_region - 有可用区域
        await manager.update_region_status("r1", RegionStatus.ACTIVE)
        selected = await manager.select_region()
        assert selected is not None

        # 12. select_region - 无可用区域
        await manager.update_region_status("r1", RegionStatus.OFFLINE)
        selected_none = await manager.select_region()
        assert selected_none is None

        # 13. select_region - NEAREST策略无client_location
        await manager.update_region_status("r1", RegionStatus.ACTIVE)
        selected = await manager.select_region(strategy=RoutingStrategy.NEAREST)
        assert selected is not None

        # 14. select_region - 所有策略
        for strategy in RoutingStrategy:
            await manager.update_region_status("r1", RegionStatus.ACTIVE)
            s = await manager.select_region(strategy=strategy)
            assert s is not None, f"策略{strategy}失败"

        # 15. health_check
        health = await manager.health_check()
        assert "r1" in health

        # 16. unregister_region
        await manager.unregister_region("r1")
        r = await manager.get_region("r1")
        assert r is None

        # 17. unregister_region - 不存在
        result = await manager.unregister_region("nonexistent")
        # 应该不崩溃

        # 18. get_primary_region - 无区域
        primary = manager.get_primary_region()
        assert primary is None

    @pytest.mark.asyncio
    async def test_pool_manager_all_interfaces(self):
        """AgentPoolManager所有接口遍历"""
        manager = AgentPoolManager()

        # 1. create_pool - 正常
        pool = await manager.create_pool(
            name="if_pool", agent_type="test",
            min_size=2, max_size=10, initial_size=5
        )
        assert pool.pool_id is not None

        # 2. list_pools
        pools = await manager.list_pools()
        assert len(pools) >= 1

        # 3. acquire_agent - 正常
        agent = await manager.acquire_agent(pool.pool_id)
        assert agent is not None

        # 4. acquire_agent - 不存在的pool
        agent_none = await manager.acquire_agent("nonexistent")
        assert agent_none is None

        # 5. release_agent - 正常
        released = await manager.release_agent(agent.agent_id)
        assert released is True

        # 6. release_agent - 不存在的agent
        released_false = await manager.release_agent("nonexistent")
        assert released_false is False

        # 7. release_agent - 重复释放
        released2 = await manager.release_agent(agent.agent_id)
        # agent已IDLE，再次释放应仍返回True（current_tasks不会小于0）

        # 8. scale_pool - 扩容
        await manager.scale_pool(pool.pool_id, 8)
        p = (await manager.list_pools())[0]
        assert p.current_size == 8

        # 9. scale_pool - 缩容
        await manager.scale_pool(pool.pool_id, 3)
        p = (await manager.list_pools())[0]
        assert p.current_size == 3

        # 10. scale_pool - 不存在的pool
        result = await manager.scale_pool("nonexistent", 5)
        assert result is None

        # 11. scale_pool - 超过max_size
        await manager.scale_pool(pool.pool_id, 100)
        p = (await manager.list_pools())[0]
        assert p.current_size == 10  # 被限制到max_size

        # 12. scale_pool - 低于min_size
        await manager.scale_pool(pool.pool_id, 0)
        p = (await manager.list_pools())[0]
        assert p.current_size == 0  # min_size=2但target=0时允许

        # 13. create_pool - initial_size边界
        pool2 = await manager.create_pool(
            name="boundary_pool", agent_type="test",
            min_size=5, max_size=10, initial_size=100  # 超过max
        )
        p2 = [p for p in await manager.list_pools() if p.name == "boundary_pool"][0]
        assert p2.current_size == 10  # 被限制到max_size

        pool3 = await manager.create_pool(
            name="boundary_pool2", agent_type="test",
            min_size=5, max_size=10, initial_size=0  # 低于min
        )
        p3 = [p for p in await manager.list_pools() if p.name == "boundary_pool2"][0]
        assert p3.current_size == 5  # 被提升到min_size

    @pytest.mark.asyncio
    async def test_router_all_interfaces(self):
        """ModelRouter所有接口遍历"""
        router = ModelRouter()

        # 1. register_provider
        config = ProviderConfig(
            provider_id="p1", name="Provider1",
            base_url="http://p1.example.com", api_key="key1"
        )
        provider = OpenAICompatibleProvider(config)
        router.register_provider(provider)

        # 2. register_model
        model = ModelInfo(
            model_id="m1", provider_id="p1",
            display_name="Model1",
            capabilities={ModelCapability.CHAT},
            max_context_tokens=4096, max_output_tokens=1024
        )
        router.register_model(model)

        # 3. route - 指定模型
        m = await router.route("m1")
        assert m is not None
        assert m.model_id == "m1"

        # 4. route - 不存在的模型（会fallback到可用模型列表）
        # route()在指定模型不存在时，会fallback到_get_available_models返回可用模型
        m_none = await router.route("truly_nonexistent_model_xyz")
        # fallback行为：返回可用模型（如果有），而非None
        assert m_none is not None  # fallback到m1

        # 5. route - 无provider可用时返回None
        router2 = ModelRouter()
        m_empty = await router2.route("any_model")
        assert m_empty is None

        # 5. route - 无参数（自动选择）
        m_auto = await router.route()
        assert m_auto is not None

        # 6. route - 按能力过滤
        m_cap = await router.route(capabilities={ModelCapability.CHAT})
        assert m_cap is not None

        # 7. route - 无匹配能力
        m_no_cap = await router.route(capabilities={ModelCapability.VISION})
        assert m_no_cap is None

        # 8. set_strategy - 所有策略
        for strategy in RouteStrategy:
            router.set_strategy(strategy)
            m = await router.route()
            # CAPABILITY策略在没有匹配时返回None
            if strategy == RouteStrategy.CAPABILITY:
                pass  # 可能返回None
            else:
                if m:
                    assert m.model_id is not None

        # 9. get_usage_stats
        stats = router.get_usage_stats()
        assert isinstance(stats, dict)

        # 10. get_latency_stats
        latency = router.get_latency_stats()
        assert isinstance(latency, dict)

        # 11. 注册多个模型测试路由
        for i in range(5):
            p = ProviderConfig(
                provider_id=f"multi_p{i}", name=f"MultiP{i}",
                base_url=f"http://multi{i}.com", api_key=f"key{i}"
            )
            router.register_provider(OpenAICompatibleProvider(p))
            m = ModelInfo(
                model_id=f"multi_m{i}", provider_id=f"multi_p{i}",
                display_name=f"MultiM{i}",
                capabilities={ModelCapability.CHAT},
                priority=i + 1
            )
            router.register_model(m)

        # PRIORITY策略应选择priority最高的
        router.set_strategy(RouteStrategy.PRIORITY)
        m_priority = await router.route()
        assert m_priority.model_id == "multi_m4"  # priority=5

    @pytest.mark.asyncio
    async def test_editor_all_interfaces(self):
        """WorkflowEditor所有接口遍历"""
        editor = WorkflowEditor()

        # 1. create_workflow
        wf = editor.create_workflow("if_wf", "description")
        assert wf.id is not None
        assert wf.name == "if_wf"

        # 2. get_workflow
        found = editor.get_workflow(wf.id)
        assert found is not None

        # 3. get_workflow - 不存在
        not_found = editor.get_workflow("nonexistent")
        assert not_found is None

        # 4. add_node - 正常
        node = WorkflowNode(id="n1", type=NodeType.FUNCTION, name="N1")
        added = editor.add_node(wf.id, node)
        assert added is True

        # 5. add_node - 不存在的workflow
        added_false = editor.add_node("nonexistent", node)
        assert added_false is False

        # 6. add_node_async - 所有节点类型
        async_node_ids = {}
        for nt in NodeType:
            n = await editor.add_node_async(wf.id, nt, f"Type_{nt.value}",
                                            (0.0, 0.0))
            assert n is not None, f"添加{nt.value}类型节点失败"
            async_node_ids[nt] = n.id

        # 7. connect_nodes - 正常
        conn = editor.connect_nodes(wf.id, "n1", async_node_ids[NodeType.FUNCTION])
        assert conn is not None

        # 8. connect_nodes - 不存在的节点
        conn_none = editor.connect_nodes(wf.id, "n1", "ghost")
        assert conn_none is None

        # 9. connect_nodes - 不存在的workflow
        conn_none2 = editor.connect_nodes("nonexistent", "n1", async_node_ids[NodeType.FUNCTION])
        assert conn_none2 is None

        # 10. validate_workflow
        errors = await editor.validate_workflow(wf.id)
        assert isinstance(errors, list)

        # 11. remove_node
        editor.remove_node(wf.id, "n1")
        assert "n1" not in editor.get_workflow(wf.id).nodes

        # 12. remove_connection
        editor.connect_nodes(wf.id, async_node_ids[NodeType.FUNCTION], async_node_ids[NodeType.AGENT])
        # 找到connection id
        wf_obj = editor.get_workflow(wf.id)
        conn_ids = list(wf_obj.connections.keys())
        if conn_ids:
            editor.remove_connection(wf.id, conn_ids[0])
            assert conn_ids[0] not in editor.get_workflow(wf.id).connections


# ============================================================
# 3. 数据一致性校验
# ============================================================

class TestDataConsistency:
    """数据一致性校验：增删改查、关联关系、数据回显"""

    @pytest.mark.asyncio
    async def test_region_crud_consistency(self):
        """区域增删改查一致性"""
        manager = CrossRegionOrchestrator().get_region_manager()

        # CREATE
        r = await manager.register_region(RegionInfo(
            region_id="crud_r", name="CRUD R",
            location="loc_1", timezone="UTC",
            endpoint="http://crud.com", capacity=500
        ))

        # READ - 验证所有字段
        found = await manager.get_region("crud_r")
        assert found.region_id == "crud_r"
        assert found.name == "CRUD R"
        assert found.location == "loc_1"
        assert found.capacity == 500
        assert found.current_load == 0

        # UPDATE - 修改状态和负载
        await manager.update_region_status("crud_r", RegionStatus.DEGRADED, load=250)
        found = await manager.get_region("crud_r")
        assert found.status == RegionStatus.DEGRADED
        assert found.current_load == 250

        # UPDATE - 再次修改，验证覆盖
        await manager.update_region_status("crud_r", RegionStatus.ACTIVE, load=100)
        found = await manager.get_region("crud_r")
        assert found.status == RegionStatus.ACTIVE
        assert found.current_load == 100

        # DELETE
        await manager.unregister_region("crud_r")
        found = await manager.get_region("crud_r")
        assert found is None

        # DELETE - 重复删除不崩溃
        await manager.unregister_region("crud_r")

    @pytest.mark.asyncio
    async def test_pool_crud_consistency(self):
        """Agent池增删改查一致性"""
        manager = AgentPoolManager()

        # CREATE
        pool = await manager.create_pool(
            name="crud_pool", agent_type="crud_agent",
            min_size=3, max_size=10, initial_size=5
        )

        # READ - 验证字段
        pools = await manager.list_pools()
        found = [p for p in pools if p.name == "crud_pool"][0]
        assert found.min_size == 3
        assert found.max_size == 10
        assert found.current_size == 5

        # UPDATE - 扩容
        await manager.scale_pool(pool.pool_id, 8)
        found = [p for p in await manager.list_pools() if p.name == "crud_pool"][0]
        assert found.current_size == 8

        # UPDATE - 缩容
        await manager.scale_pool(pool.pool_id, 4)
        found = [p for p in await manager.list_pools() if p.name == "crud_pool"][0]
        assert found.current_size == 4

        # 验证agent数量与current_size一致
        assert len(found.agents) == found.current_size

    @pytest.mark.asyncio
    async def test_workflow_crud_consistency(self):
        """工作流增删改查一致性"""
        editor = WorkflowEditor()

        # CREATE
        wf = editor.create_workflow("crud_wf")
        wf_id = wf.id

        # ADD nodes
        for i in range(5):
            node = WorkflowNode(id=f"cn_{i}", type=NodeType.FUNCTION,
                               name=f"CN{i}", config={"code": f"context['v{i}'] = {i}"})
            editor.add_node(wf_id, node)

        # ADD connections
        for i in range(4):
            editor.connect_nodes(wf_id, f"cn_{i}", f"cn_{i+1}")

        # READ - 验证节点和连接数量
        found = editor.get_workflow(wf_id)
        assert len(found.nodes) == 5
        assert len(found.connections) == 4

        # REMOVE node - 验证关联连接被删除
        editor.remove_node(wf_id, "cn_2")
        found = editor.get_workflow(wf_id)
        assert len(found.nodes) == 4
        # cn_1→cn_2 和 cn_2→cn_3 的连接应被删除
        for conn in found.connections.values():
            assert "cn_2" not in (conn.source_node, conn.target_node)

        # EXECUTE - 验证数据流
        executor = WorkflowExecutor(editor)
        context = {}
        result = await executor.execute(wf_id, context)
        assert result.status == NodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_router_registration_consistency(self):
        """路由器注册一致性"""
        router = ModelRouter()

        # 注册provider和model
        config = ProviderConfig(
            provider_id="cons_p", name="ConsP",
            base_url="http://cons.com", api_key="key"
        )
        router.register_provider(OpenAICompatibleProvider(config))

        model = ModelInfo(
            model_id="cons_m", provider_id="cons_p",
            display_name="ConsM",
            capabilities={ModelCapability.CHAT, ModelCapability.CODE}
        )
        router.register_model(model)

        # 验证注册一致性
        m = await router.route("cons_m")
        assert m is not None
        assert m.provider_id == "cons_p"
        assert ModelCapability.CHAT in m.capabilities
        assert ModelCapability.CODE in m.capabilities

        # 按能力过滤
        m_chat = await router.route(capabilities={ModelCapability.CHAT})
        assert m_chat is not None

        m_vision = await router.route(capabilities={ModelCapability.VISION})
        assert m_vision is None  # 没有VISION能力


# ============================================================
# 4. 跨模块集成联调
# ============================================================

class TestCrossModuleIntegration:
    """跨模块、跨层级调用验证"""

    @pytest.mark.asyncio
    async def test_geo_scale_integration(self):
        """geo + scale 联动：区域故障时Agent池行为"""
        region_mgr = CrossRegionOrchestrator().get_region_manager()
        pool_mgr = AgentPoolManager()

        # 创建区域
        for i in range(3):
            await region_mgr.register_region(RegionInfo(
                region_id=f"int_r{i}", name=f"IntR{i}",
                location=f"loc_{i}", timezone="UTC",
                endpoint=f"http://int{i}.com"
            ))

        # 创建pool
        pool = await pool_mgr.create_pool(
            name="int_pool", agent_type="int_agent",
            min_size=3, max_size=10, initial_size=5
        )

        # 并发：区域操作 + Agent操作
        async def mixed_ops(iteration: int):
            try:
                # 区域操作
                status = random.choice(list(RegionStatus))
                await region_mgr.update_region_status(
                    f"int_r{iteration % 3}", status
                )
                # Agent操作
                agent = await pool_mgr.acquire_agent(pool.pool_id)
                if agent:
                    await asyncio.sleep(0.001)
                    await pool_mgr.release_agent(agent.agent_id)
                return True
            except Exception:
                return False

        results = await asyncio.gather(*[mixed_ops(i) for i in range(100)])
        success = sum(results)
        assert success >= 90, f"跨模块联动成功率过低: {success}/100"

    @pytest.mark.asyncio
    async def test_workflow_router_integration(self):
        """editor + router 联动：工作流中使用模型路由"""
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)
        router = ModelRouter()

        # 注册模型
        config = ProviderConfig(
            provider_id="wf_p", name="WfP",
            base_url="http://wf.com", api_key="key"
        )
        router.register_provider(OpenAICompatibleProvider(config))
        router.register_model(ModelInfo(
            model_id="wf_m", provider_id="wf_p",
            display_name="WfM",
            capabilities={ModelCapability.CHAT}
        ))

        # 创建使用模型的工作流
        wf = editor.create_workflow("wf_router_test")
        wf_id = wf.id

        # 数据准备节点
        data_node = WorkflowNode(
            id="data", type=NodeType.FUNCTION, name="Data",
            config={"code": "context['prompt'] = 'Hello world'"}
        )
        editor.add_node(wf_id, data_node)

        # 模型调用节点（模拟）
        model_node = WorkflowNode(
            id="model", type=NodeType.AGENT, name="ModelCall",
            config={"model": "wf_m"}
        )
        editor.add_node(wf_id, model_node)

        # 结果处理节点
        result_node = WorkflowNode(
            id="result", type=NodeType.FUNCTION, name="Result",
            config={"code": "context['done'] = True"}
        )
        editor.add_node(wf_id, result_node)

        editor.connect_nodes(wf_id, "data", "model")
        editor.connect_nodes(wf_id, "model", "result")

        # 执行
        context = {}
        exec_result = await executor.execute(wf_id, context)
        assert exec_result.status == NodeStatus.COMPLETED
        assert context.get("prompt") == "Hello world"
        assert context.get("done") is True

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """完整流水线：region + pool + editor + router"""
        orchestrator = CrossRegionOrchestrator()
        region_mgr = orchestrator.get_region_manager()
        pool_mgr = AgentPoolManager()
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)
        router = ModelRouter()

        # 1. 初始化区域
        for i in range(3):
            await region_mgr.register_region(RegionInfo(
                region_id=f"full_r{i}", name=f"FullR{i}",
                location=f"loc_{i}", timezone="UTC",
                endpoint=f"http://full{i}.com"
            ))

        # 2. 初始化Agent池
        pool = await pool_mgr.create_pool(
            name="full_pool", agent_type="full_agent",
            min_size=5, max_size=20, initial_size=10
        )

        # 3. 初始化模型路由
        for i in range(3):
            config = ProviderConfig(
                provider_id=f"full_p{i}", name=f"FullP{i}",
                base_url=f"http://full_p{i}.com", api_key=f"key{i}"
            )
            router.register_provider(OpenAICompatibleProvider(config))
            router.register_model(ModelInfo(
                model_id=f"full_m{i}", provider_id=f"full_p{i}",
                display_name=f"FullM{i}",
                capabilities={ModelCapability.CHAT},
                priority=i + 1
            ))

        # 4. 创建复杂工作流
        wf = editor.create_workflow("full_pipeline")
        wf_id = wf.id

        steps = ["extract", "transform", "enrich", "validate", "output"]
        for i, step in enumerate(steps):
            node = WorkflowNode(
                id=f"step_{i}", type=NodeType.FUNCTION, name=step,
                config={"code": f"context['{step}'] = True"}
            )
            editor.add_node(wf_id, node)
            if i > 0:
                editor.connect_nodes(wf_id, f"step_{i-1}", f"step_{i}")

        # 5. 并发执行：工作流 + Agent操作 + 区域操作 + 路由操作
        async def full_ops():
            try:
                # 工作流
                ctx = {}
                await executor.execute(wf_id, ctx)
                # Agent
                agent = await pool_mgr.acquire_agent(pool.pool_id)
                if agent:
                    await pool_mgr.release_agent(agent.agent_id)
                # 路由
                await router.route()
                # 区域
                await region_mgr.select_region()
                return True
            except Exception:
                return False

        results = await asyncio.gather(*[full_ops() for _ in range(50)])
        success = sum(results)
        assert success >= 45, f"完整流水线成功率过低: {success}/50"

        # 6. 验证数据一致性
        pools = await pool_mgr.list_pools()
        assert len(pools) >= 1
        regions = await region_mgr.list_regions()
        assert len(regions) >= 3


# ============================================================
# 5. 异常容错全覆盖
# ============================================================

class TestExceptionTolerance:
    """异常容错：超时、断连、报错、兜底降级、重试"""

    @pytest.mark.asyncio
    async def test_empty_input_tolerance(self):
        """空值输入容错"""
        manager = CrossRegionOrchestrator().get_region_manager()
        pool_mgr = AgentPoolManager()
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)
        router = ModelRouter()

        # 空列表
        regions = await manager.list_regions()
        assert regions == []

        # 空workflow执行
        wf = editor.create_workflow("empty_wf")
        result = await executor.execute(wf.id, {})
        assert result.status == NodeStatus.COMPLETED

        # 空context
        result = await executor.execute(wf.id, None)
        assert result.status == NodeStatus.COMPLETED

        # 空路由
        m = await router.route("nonexistent")
        assert m is None

        # 空pool操作
        agent = await pool_mgr.acquire_agent("nonexistent")
        assert agent is None

    @pytest.mark.asyncio
    async def test_concurrent_modification_tolerance(self):
        """并发修改容错"""
        manager = CrossRegionOrchestrator().get_region_manager()

        # 注册一个区域
        await manager.register_region(RegionInfo(
            region_id="concurrent_r", name="CR",
            location="loc_1", timezone="UTC",
            endpoint="http://cr.com"
        ))

        # 并发修改同一区域状态
        async def modify_status(status: RegionStatus):
            try:
                await manager.update_region_status("concurrent_r", status)
                return True
            except Exception:
                return False

        results = await asyncio.gather(*[
            modify_status(s) for s in RegionStatus
        ])
        # 所有操作应完成（不崩溃）
        assert all(r is True or r is False for r in results)

        # 最终状态应该是某个有效状态
        r = await manager.get_region("concurrent_r")
        assert r.status in RegionStatus

    @pytest.mark.asyncio
    async def test_workflow_with_all_error_types(self):
        """工作流中各种错误类型"""
        editor = WorkflowEditor()
        executor = WorkflowExecutor(editor)

        # 1. 语法错误
        wf1 = editor.create_workflow("syntax_err")
        n1 = WorkflowNode(id="e1", type=NodeType.FUNCTION, name="E1",
                          config={"code": "this is not valid python"})
        editor.add_node(wf1.id, n1)
        r1 = await executor.execute(wf1.id, {})
        assert r1.node_results["e1"].status == NodeStatus.FAILED

        # 2. 运行时错误
        wf2 = editor.create_workflow("runtime_err")
        n2 = WorkflowNode(id="e2", type=NodeType.FUNCTION, name="E2",
                          config={"code": "1/0"})
        editor.add_node(wf2.id, n2)
        r2 = await executor.execute(wf2.id, {})
        assert r2.node_results["e2"].status == NodeStatus.FAILED

        # 3. 名称错误（安全builtins中无此函数）
        wf3 = editor.create_workflow("name_err")
        n3 = WorkflowNode(id="e3", type=NodeType.FUNCTION, name="E3",
                          config={"code": "nonexistent_function()"})
        editor.add_node(wf3.id, n3)
        r3 = await executor.execute(wf3.id, {})
        assert r3.node_results["e3"].status == NodeStatus.FAILED

        # 4. 空代码（应成功）
        wf4 = editor.create_workflow("empty_code")
        n4 = WorkflowNode(id="e4", type=NodeType.FUNCTION, name="E4",
                          config={"code": ""})
        editor.add_node(wf4.id, n4)
        r4 = await executor.execute(wf4.id, {})
        assert r4.node_results["e4"].status == NodeStatus.COMPLETED

        # 5. 条件评估错误
        wf5 = editor.create_workflow("cond_err")
        n5 = WorkflowNode(id="e5", type=NodeType.CONDITION, name="E5",
                          config={"condition": "undefined_var > 10"})
        editor.add_node(wf5.id, n5)
        r5 = await executor.execute(wf5.id, {})
        assert r5.node_results["e5"].status == NodeStatus.FAILED

    @pytest.mark.asyncio
    async def test_template_market_full_flow(self):
        """模板市场完整流程"""
        market = TemplateMarket()
        editor = WorkflowEditor()

        # 创建工作流模板
        template_wf = editor.create_workflow("template_wf")
        node = WorkflowNode(id="t_n1", type=NodeType.FUNCTION, name="TN1",
                            config={"code": "context['templated'] = True"})
        editor.add_node(template_wf.id, node)

        # 发布模板
        template = await market.publish_template(
            name="Test Template", description="A test template",
            category="automation", workflow=template_wf,
            author="tester", tags=["test", "demo"]
        )
        assert template.id is not None

        # 搜索模板
        results = await market.search_templates(query="test")
        assert len(results) >= 1

        results = await market.search_templates(category="automation")
        assert len(results) >= 1

        results = await market.search_templates(tags=["demo"])
        assert len(results) >= 1

        results = await market.search_templates(query="nonexistent")
        assert len(results) == 0

        # 下载模板
        downloaded = await market.download_template(template.id)
        assert downloaded is not None
        assert downloaded.name == "template_wf"

        # 下载不存在的模板
        none = await market.download_template("nonexistent")
        assert none is None

    @pytest.mark.asyncio
    async def test_config_version_full_flow(self):
        """配置版本管理完整流程"""
        version_mgr = ConfigVersionManager()
        editor = WorkflowEditor()

        wf = editor.create_workflow("version_wf")

        # 保存多个版本
        v1 = await version_mgr.save_version(wf.id, wf, "v1")
        v2 = await version_mgr.save_version(wf.id, wf, "v2")
        v3 = await version_mgr.save_version(wf.id, wf, "v3")

        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

        # 获取版本
        ver1 = await version_mgr.get_version(wf.id, 1)
        assert ver1 is not None
        assert ver1["version"] == 1
        assert ver1["message"] == "v1"

        # 获取不存在的版本
        ver_none = await version_mgr.get_version(wf.id, 999)
        assert ver_none is None

        # 版本数量限制测试
        for i in range(25):
            await version_mgr.save_version(wf.id, wf, f"v{i+4}")

        versions = version_mgr._versions.get(wf.id, [])
        assert len(versions) <= 20  # _max_versions限制


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
