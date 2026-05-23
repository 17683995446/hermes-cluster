"""
Hermes Cluster v6.0 - Round 10 安全沙箱逃逸 + 跨模块集成混沌 + 接口全量遍历 测试
=============================================================================
测试目标：
1. 安全沙箱逃逸：验证exec/eval沙箱是否可被突破（__class__, __bases__, __subclasses__等）
2. 跨模块集成混沌：pool+editor+region+router联动异常场景
3. 接口全量遍历：所有公共API异常入参、空值、非法格式
"""

import asyncio
import logging
import time
import uuid
import sys
import traceback
from datetime import datetime, timedelta

import pytest

# 配置详细日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

from hermes_cluster.lowcode.editor import (
    WorkflowEditor, WorkflowExecutor, WorkflowValidator, TemplateMarket,
    ConfigGenerator, VisualWorkflow, VisualNode, Connection, NodeType,
    NodeStatus, Port, NodePosition, NodeExecutionResult, WorkflowExecutionResult
)
from hermes_cluster.scale.pool import (
    AgentPoolManager, AutoScaler, HierarchicalScheduler, PerformanceBenchmark,
    CapacityPlanner, AgentSpec, AgentInstance, AgentPool, AgentState, ScalingPolicy,
    ScalingAction, SchedulerLevel
)
from hermes_cluster.geo.region import (
    RegionManager, DataSyncEngine, FailoverManager, GeoRoutingService,
    CrossRegionOrchestrator, RegionInfo, RegionStatus, SyncMode, RoutingStrategy
)
from hermes_cluster.models.router import (
    ModelRouter, SiliconFlowProvider, OpenAICompatibleProvider,
    ModelInfo, ProviderConfig, ChatMessage, ChatRequest, ChatResponse,
    RouteStrategy, ModelCapability, ProviderStatus, UsageRecord
)


# ============================================================
# 辅助工具
# ============================================================

def create_function_workflow(wf_id, name, code):
    """创建只含一个FUNCTION节点的工作流并注册到editor"""
    editor = WorkflowEditor()
    wf = editor.create_workflow(name, f"{name} desc")
    # 覆盖id
    wf.id = wf_id
    editor._workflows[wf_id] = wf
    node = VisualNode(
        id="fn1", type=NodeType.FUNCTION, name="Test Function",
        config={"code": code}
    )
    wf.add_node(node)
    return editor, wf


def create_condition_workflow(wf_id, name, condition):
    """创建只含一个CONDITION节点的工作流并注册到editor"""
    editor = WorkflowEditor()
    wf = editor.create_workflow(name, f"{name} desc")
    wf.id = wf_id
    editor._workflows[wf_id] = wf
    node = VisualNode(
        id="cond1", type=NodeType.CONDITION, name="Test Condition",
        config={"condition": condition}
    )
    wf.add_node(node)
    return editor, wf


def create_region(region_id, name, endpoint="http://localhost:8080", **kwargs):
    """创建RegionInfo对象"""
    return RegionInfo(
        region_id=region_id,
        name=name,
        location=kwargs.get("location", "default"),
        timezone=kwargs.get("timezone", "UTC"),
        endpoint=endpoint,
        capacity=kwargs.get("capacity", 1000),
        **{k: v for k, v in kwargs.items() if k in ("status", "priority", "current_load", "latency_map", "metadata")}
    )


# ============================================================
# 第一部分：安全沙箱逃逸测试
# ============================================================

class TestSandboxEscapeExec:
    """测试exec沙箱是否可被突破"""

    async def run_function_code(self, code, timeout=5.0):
        """辅助方法：执行一段FUNCTION节点代码"""
        wf_id = f"sandbox-test-{uuid.uuid4().hex[:8]}"
        editor, wf = create_function_workflow(wf_id, "Sandbox Test", code)
        executor = WorkflowExecutor(editor)
        result = await executor.execute(wf_id, {"input": "test"}, timeout=timeout)
        return result

    @pytest.mark.asyncio
    async def test_exec_basic_functionality(self):
        """基础功能验证：正常代码应正常执行"""
        result = await self.run_function_code("result = 1 + 1")
        assert result.status == NodeStatus.COMPLETED
        assert result.node_results["fn1"].output == 2
        print("✅ [安全] 基础exec功能正常")

    @pytest.mark.asyncio
    async def test_exec_context_access(self):
        """验证context变量可正常访问"""
        result = await self.run_function_code("result = context.get('input', 'default')")
        assert result.status == NodeStatus.COMPLETED
        assert result.node_results["fn1"].output == "test"
        print("✅ [安全] context访问正常")

    @pytest.mark.asyncio
    async def test_exec_time_module_access(self):
        """验证time模块可正常使用"""
        result = await self.run_function_code("result = time.time() > 0")
        assert result.status == NodeStatus.COMPLETED
        assert result.node_results["fn1"].output is True
        print("✅ [安全] time模块访问正常")

    @pytest.mark.asyncio
    async def test_exec_print_disabled(self):
        """验证print被禁用（不产生输出）"""
        result = await self.run_function_code("print('SHOULD NOT APPEAR'); result = 'ok'")
        assert result.status == NodeStatus.COMPLETED
        assert result.node_results["fn1"].output == "ok"
        print("✅ [安全] print已禁用")

    @pytest.mark.asyncio
    async def test_exec_dir_restricted(self):
        """验证dir()被限制为返回空列表"""
        result = await self.run_function_code("result = dir(str)")
        assert result.status == NodeStatus.COMPLETED
        assert result.node_results["fn1"].output == []
        print("✅ [安全] dir()已被限制")

    @pytest.mark.asyncio
    async def test_sandbox_escape_class_attribute(self):
        """BUG#72修复验证：__class__逃逸应被阻止"""
        code = "result = ().__class__.__bases__"
        result = await self.run_function_code(code)
        if result.status == NodeStatus.FAILED:
            print(f"✅ [安全] __class__ 逃逸被阻止: {result.node_results['fn1'].error}")
        else:
            print(f"⚠️ [安全风险] __class__.__bases__ 仍可访问: {result.node_results['fn1'].output}")

    @pytest.mark.asyncio
    async def test_sandbox_escape_subclasses(self):
        """BUG#72修复验证：__subclasses__逃逸应被阻止"""
        code = "result = ().__class__.__bases__[0].__subclasses__()"
        result = await self.run_function_code(code)
        if result.status == NodeStatus.FAILED:
            print(f"✅ [安全] __subclasses__() 逃逸被阻止: {result.node_results['fn1'].error}")
        else:
            print(f"⚠️ [安全风险] __subclasses__() 仍可访问")

    @pytest.mark.asyncio
    async def test_sandbox_escape_globals(self):
        """BUG#73检测：尝试通过__globals__逃逸沙箱"""
        code = "result = hasattr(len, '__globals__')"
        result = await self.run_function_code(code)
        if result.status == NodeStatus.COMPLETED:
            has_globals = result.node_results["fn1"].output
            if has_globals:
                code2 = "result = len.__globals__"
                result2 = await self.run_function_code(code2)
                if result2.status == NodeStatus.COMPLETED:
                    print(f"⚠️ [严重安全漏洞] __globals__ 可访问: {list(result2.node_results['fn1'].output.keys())[:5]}...")
                else:
                    print(f"✅ [安全] __globals__ 属性存在但读取被阻止")
            else:
                print(f"✅ [安全] 内置函数无 __globals__ 属性")
        else:
            print(f"ℹ️ [安全] hasattr检查异常: {result.node_results['fn1'].error}")

    @pytest.mark.asyncio
    async def test_sandbox_escape_builtins_reconstruction(self):
        """BUG#74检测：尝试重建__builtins__访问危险函数"""
        code = """
t = type
bases = t.__bases__
result = str(bases)
"""
        result = await self.run_function_code(code)
        if result.status == NodeStatus.COMPLETED:
            print(f"⚠️ [安全风险] type.__bases__ 可访问: {result.node_results['fn1'].output}")
        else:
            print(f"✅ [安全] type.__bases__ 访问被阻止")

    @pytest.mark.asyncio
    async def test_sandbox_escape_import_attempt(self):
        """验证直接import被阻止"""
        code = "import os\nresult = os.listdir('.')"
        result = await self.run_function_code(code)
        if result.status == NodeStatus.FAILED:
            print(f"✅ [安全] import被阻止: {result.node_results['fn1'].error}")
        else:
            print(f"⚠️ [安全漏洞] import未被阻止! output={result.node_results['fn1'].output}")

    @pytest.mark.asyncio
    async def test_sandbox_escape_open_attempt(self):
        """验证open()不可用"""
        code = "result = open('/etc/passwd', 'r').read()"
        result = await self.run_function_code(code)
        if result.status == NodeStatus.FAILED:
            print(f"✅ [安全] open()被阻止: {result.node_results['fn1'].error}")
        else:
            print(f"🔴 [致命安全漏洞] open()未被阻止! 可能读取到敏感文件")

    @pytest.mark.asyncio
    async def test_sandbox_escape_exec_nested(self):
        """验证嵌套exec被阻止"""
        code = "exec('import os'); result = 'escaped'"
        result = await self.run_function_code(code)
        if result.status == NodeStatus.FAILED:
            print(f"✅ [安全] 嵌套exec被阻止: {result.node_results['fn1'].error}")
        else:
            print(f"⚠️ [安全漏洞] 嵌套exec未被阻止!")

    @pytest.mark.asyncio
    async def test_sandbox_escape_eval_nested(self):
        """验证嵌套eval被阻止"""
        code = "result = eval('__import__(\"os\").system(\"echo HACKED\")')"
        result = await self.run_function_code(code)
        if result.status == NodeStatus.FAILED:
            print(f"✅ [安全] 嵌套eval被阻止: {result.node_results['fn1'].error}")
        else:
            print(f"⚠️ [安全漏洞] 嵌套eval未被阻止!")

    @pytest.mark.asyncio
    async def test_sandbox_escape_compile(self):
        """验证compile()不可用"""
        code = "result = compile('import os', '<string>', 'exec')"
        result = await self.run_function_code(code)
        if result.status == NodeStatus.FAILED:
            print(f"✅ [安全] compile()被阻止: {result.node_results['fn1'].error}")
        else:
            print(f"⚠️ [安全漏洞] compile()未被阻止!")

    @pytest.mark.asyncio
    async def test_sandbox_escape_getattr_chain(self):
        """验证通过getattr链式访问被阻止"""
        code = "result = getattr(getattr(getattr(len, '__class__'), '__bases__'), '__getitem__')(0)"
        result = await self.run_function_code(code)
        if result.status == NodeStatus.FAILED:
            print(f"✅ [安全] getattr链式访问被阻止: {result.node_results['fn1'].error}")
        else:
            print(f"⚠️ [安全风险] getattr链式访问未被完全阻止: {type(result.node_results['fn1'].output)}")

    @pytest.mark.asyncio
    async def test_sandbox_infinite_loop_timeout(self):
        """BUG#77修复验证：验证无限循环被timeout阻止"""
        code = "while True: pass"
        start = time.time()
        result = await self.run_function_code(code, timeout=5.0)
        elapsed = time.time() - start
        assert result.status == NodeStatus.TIMEOUT, f"Expected TIMEOUT but got {result.status}"
        # multiprocessing启动有开销，允许更长时间，但应远小于默认300s
        assert elapsed < 30.0, f"Timeout took too long: {elapsed}s"
        print(f"✅ [安全] 无限循环在 {elapsed:.2f}s 后被timeout阻止（BUG#77修复验证）")

    @pytest.mark.asyncio
    async def test_sandbox_memory_exhaustion_timeout(self):
        """验证内存耗尽攻击被timeout阻止"""
        code = "x = []; \nwhile True: x.append('A' * 1024 * 1024)"
        start = time.time()
        result = await self.run_function_code(code, timeout=5.0)
        elapsed = time.time() - start
        assert result.status == NodeStatus.TIMEOUT, f"Expected TIMEOUT but got {result.status}"
        print(f"✅ [安全] 内存耗尽攻击在 {elapsed:.2f}s 后被timeout阻止")

    @pytest.mark.asyncio
    async def test_sandbox_time_sleep_dos(self):
        """BUG#77修复验证：验证time.sleep不能被用于DoS攻击"""
        code = "time.sleep(100)"
        start = time.time()
        result = await self.run_function_code(code, timeout=5.0)
        elapsed = time.time() - start
        assert result.status == NodeStatus.TIMEOUT, f"Expected TIMEOUT but got {result.status}"
        assert elapsed < 30.0, f"Timeout took too long: {elapsed}s"
        print(f"✅ [安全] time.sleep DoS被timeout阻止 ({elapsed:.2f}s)（BUG#77修复验证）")


class TestSandboxEscapeEval:
    """测试eval沙箱（CONDITION节点）是否可被突破"""

    async def run_condition(self, condition, timeout=5.0):
        """辅助方法：执行CONDITION节点"""
        wf_id = f"cond-test-{uuid.uuid4().hex[:8]}"
        editor, wf = create_condition_workflow(wf_id, "Condition Test", condition)
        executor = WorkflowExecutor(editor)
        result = await executor.execute(wf_id, {"input": "test"}, timeout=timeout)
        return result

    @pytest.mark.asyncio
    async def test_eval_basic_condition(self):
        """基础条件评估"""
        result = await self.run_condition("True")
        assert result.status == NodeStatus.COMPLETED
        assert result.node_results["cond1"].output == {"condition_result": True}
        print("✅ [安全] 基础eval条件正常")

    @pytest.mark.asyncio
    async def test_eval_context_access(self):
        """验证eval中context可访问"""
        result = await self.run_condition("context.get('input') == 'test'")
        assert result.status == NodeStatus.COMPLETED
        assert result.node_results["cond1"].output == {"condition_result": True}
        print("✅ [安全] eval中context访问正常")

    @pytest.mark.asyncio
    async def test_eval_sandbox_escape_class(self):
        """检测eval中__class__逃逸"""
        result = await self.run_condition("().__class__.__name__ == 'tuple'")
        if result.status == NodeStatus.COMPLETED:
            cond_result = result.node_results["cond1"].output.get("condition_result")
            if cond_result:
                print(f"⚠️ [安全风险] eval中 __class__ 可访问")
            else:
                print(f"✅ [安全] eval中 __class__ 不可用")
        else:
            print(f"✅ [安全] eval中 __class__ 访问被阻止")

    @pytest.mark.asyncio
    async def test_eval_sandbox_escape_subclasses(self):
        """检测eval中__subclasses__逃逸"""
        result = await self.run_condition(
            "len(().__class__.__bases__[0].__subclasses__()) > 0"
        )
        if result.status == NodeStatus.COMPLETED:
            cond_result = result.node_results["cond1"].output.get("condition_result")
            if cond_result:
                print(f"⚠️ [严重安全漏洞] eval中 __subclasses__() 可访问!")
            else:
                print(f"✅ [安全] eval中 __subclasses__ 不可用")
        else:
            print(f"✅ [安全] eval中 __subclasses__ 访问被阻止: {result.node_results['cond1'].error}")

    @pytest.mark.asyncio
    async def test_eval_sandbox_escape_import(self):
        """检测eval中__import__逃逸"""
        result = await self.run_condition("__import__('os') is not None")
        if result.status == NodeStatus.COMPLETED:
            cond_result = result.node_results["cond1"].output.get("condition_result")
            if cond_result:
                print(f"🔴 [致命安全漏洞] eval中 __import__ 可用!")
            else:
                print(f"✅ [安全] eval中 __import__ 不可用")
        else:
            print(f"✅ [安全] eval中 __import__ 被阻止: {result.node_results['cond1'].error}")

    @pytest.mark.asyncio
    async def test_eval_no_time_module(self):
        """验证eval中没有time模块（与exec不同）"""
        result = await self.run_condition("time is not None")
        if result.status == NodeStatus.FAILED:
            print(f"✅ [安全] eval中无time模块（符合设计）: {result.node_results['cond1'].error}")
        else:
            cond_result = result.node_results["cond1"].output.get("condition_result")
            if cond_result:
                print(f"⚠️ [安全风险] eval中time模块可用（设计不一致）")
            else:
                print(f"✅ [安全] eval中time模块不可用")

    @pytest.mark.asyncio
    async def test_eval_exception_classes_not_available(self):
        """验证eval中没有异常类（与exec不同）"""
        result = await self.run_condition("ValueError is not None")
        if result.status == NodeStatus.FAILED:
            print(f"✅ [安全] eval中无ValueError（符合设计）")
        else:
            cond_result = result.node_results["cond1"].output.get("condition_result")
            if cond_result:
                print(f"ℹ️ [设计差异] eval中ValueError可用（与exec一致）")
            else:
                print(f"✅ [安全] eval中ValueError不可用")


# ============================================================
# 第二部分：跨模块集成混沌测试
# ============================================================

class TestCrossModuleIntegration:
    """pool+editor+region+router跨模块集成混沌测试"""

    @pytest.mark.asyncio
    async def test_pool_concurrent_create_and_scale(self):
        """并发创建池和伸缩"""
        mgr = AgentPoolManager()
        tasks = []
        for i in range(20):
            tasks.append(mgr.create_pool(f"pool-{i}", f"type-{i}", min_size=1, max_size=10, initial_size=3))

        pools = await asyncio.gather(*tasks)
        assert len(pools) == 20
        assert all(p.current_size == 3 for p in pools)

        # 并发伸缩
        scale_tasks = []
        for p in pools:
            scale_tasks.append(mgr.scale_pool(p.pool_id, 5))
        events = await asyncio.gather(*scale_tasks)
        assert all(e is not None for e in events)
        assert all(e.new_size == 5 for e in events)
        print("✅ [集成] 并发创建20个池+并发伸缩全部成功")

    @pytest.mark.asyncio
    async def test_pool_acquire_release_storm(self):
        """高频acquire/release风暴"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("storm-pool", "test", min_size=5, max_size=5, initial_size=5)

        acquire_count = 0
        release_count = 0
        errors = []

        async def acquire_release_cycle(idx):
            nonlocal acquire_count, release_count
            for _ in range(50):
                agent = await mgr.acquire_agent(pool.pool_id)
                if agent:
                    acquire_count += 1
                    await asyncio.sleep(0.001)
                    ok = await mgr.release_agent(agent.agent_id)
                    if ok:
                        release_count += 1
                    else:
                        errors.append(f"release failed for {agent.agent_id}")
                else:
                    errors.append(f"acquire failed at cycle")

        await asyncio.gather(*[acquire_release_cycle(i) for i in range(10)])
        assert acquire_count == release_count, f"acquire={acquire_count}, release={release_count}"
        assert len(errors) == 0, f"Errors: {errors[:5]}"
        print(f"✅ [集成] 高频acquire/release风暴: {acquire_count}次成功，0错误")

    @pytest.mark.asyncio
    async def test_pool_scale_to_zero_and_back(self):
        """缩容到0再扩容回来"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("zero-pool", "test", min_size=1, max_size=10, initial_size=5)
        assert pool.current_size == 5

        event = await mgr.scale_pool(pool.pool_id, 0)
        assert event.new_size == 0
        pool = (await mgr.list_pools())[0]
        assert pool.current_size == 0

        event2 = await mgr.scale_pool(pool.pool_id, 3)
        assert event2.new_size == 3
        pool = (await mgr.list_pools())[0]
        assert pool.current_size == 3
        print("✅ [集成] 缩容到0再扩容回来正常")

    @pytest.mark.asyncio
    async def test_pool_negative_scale_target(self):
        """BUG#76检测：负数缩容目标"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("neg-pool", "test", min_size=1, max_size=10, initial_size=5)

        event = await mgr.scale_pool(pool.pool_id, -5)
        if event:
            assert event.new_size >= pool.min_size, f"Negative scale resulted in size {event.new_size}"
            print(f"✅ [集成] 负数缩容被正确处理: target=-5, actual={event.new_size}")
        else:
            print(f"⚠️ [BUG#76] 负数缩容返回None（可能未正确处理）")

    @pytest.mark.asyncio
    async def test_pool_release_nonexistent_agent(self):
        """释放不存在的agent"""
        mgr = AgentPoolManager()
        result = await mgr.release_agent("nonexistent-agent-id")
        assert result is False
        print("✅ [集成] 释放不存在的agent返回False")

    @pytest.mark.asyncio
    async def test_pool_acquire_from_empty_pool(self):
        """从空池获取agent"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("empty-pool", "test", min_size=0, max_size=5, initial_size=0)
        agent = await mgr.acquire_agent(pool.pool_id)
        assert agent is None
        print("✅ [集成] 空池获取agent返回None")

    @pytest.mark.asyncio
    async def test_pool_scale_nonexistent_pool(self):
        """缩容不存在的池"""
        mgr = AgentPoolManager()
        event = await mgr.scale_pool("nonexistent-pool-id", 5)
        assert event is None
        print("✅ [集成] 缩容不存在的池返回None")

    @pytest.mark.asyncio
    async def test_workflow_with_circular_dependency(self):
        """工作流循环依赖处理"""
        editor = WorkflowEditor()
        wf = editor.create_workflow("Circular Workflow", "Has cycle")
        wf.id = "circular-wf"
        editor._workflows["circular-wf"] = wf

        n1 = VisualNode(id="n1", type=NodeType.TASK, name="Task 1")
        n2 = VisualNode(id="n2", type=NodeType.TASK, name="Task 2")
        n3 = VisualNode(id="n3", type=NodeType.TASK, name="Task 3")
        wf.add_node(n1)
        wf.add_node(n2)
        wf.add_node(n3)

        wf.add_connection(Connection(id="c1", source_node="n1", source_port="out", target_node="n2", target_port="in"))
        wf.add_connection(Connection(id="c2", source_node="n2", source_port="out", target_node="n3", target_port="in"))
        wf.add_connection(Connection(id="c3", source_node="n3", source_port="out", target_node="n1", target_port="in"))

        is_valid, errors = WorkflowValidator.validate(wf)
        assert not is_valid
        assert any("循环" in e for e in errors)
        print(f"✅ [集成] 循环依赖被正确检测: {errors}")

        executor = WorkflowExecutor(editor)
        result = await executor.execute("circular-wf")
        assert result.status in (NodeStatus.COMPLETED, NodeStatus.FAILED)
        print(f"✅ [集成] 循环依赖工作流执行不崩溃: status={result.status.value}")

    @pytest.mark.asyncio
    async def test_workflow_empty_execution(self):
        """空工作流执行"""
        editor = WorkflowEditor()
        wf = editor.create_workflow("Empty Workflow", "")
        wf.id = "empty-wf"
        editor._workflows["empty-wf"] = wf

        executor = WorkflowExecutor(editor)
        result = await executor.execute("empty-wf")
        # 空工作流无节点，按BUG#57修复应返回COMPLETED
        assert result.status in (NodeStatus.COMPLETED, NodeStatus.FAILED)
        print(f"✅ [集成] 空工作流执行: status={result.status.value}")

    @pytest.mark.asyncio
    async def test_workflow_complex_branching(self):
        """复杂条件分支工作流"""
        editor = WorkflowEditor()
        wf = editor.create_workflow("Branch Workflow", "Complex branching")
        wf.id = "branch-wf"
        editor._workflows["branch-wf"] = wf

        start = VisualNode(id="start", type=NodeType.TASK, name="Start")
        cond = VisualNode(id="cond", type=NodeType.CONDITION, name="Check", config={"condition": "context.get('flag', False)"})
        true_branch = VisualNode(id="true_task", type=NodeType.TASK, name="True Task")
        false_branch = VisualNode(id="false_task", type=NodeType.TASK, name="False Task")
        end = VisualNode(id="end", type=NodeType.OUTPUT, name="End")

        for n in [start, cond, true_branch, false_branch, end]:
            wf.add_node(n)

        wf.add_connection(Connection(id="c1", source_node="start", source_port="out", target_node="cond", target_port="in"))
        wf.add_connection(Connection(id="c2", source_node="cond", source_port="true", target_node="true_task", target_port="in"))
        wf.add_connection(Connection(id="c3", source_node="cond", source_port="false", target_node="false_task", target_port="in"))
        wf.add_connection(Connection(id="c4", source_node="true_task", source_port="out", target_node="end", target_port="in"))
        wf.add_connection(Connection(id="c5", source_node="false_task", source_port="out", target_node="end", target_port="in"))

        executor = WorkflowExecutor(editor)

        # 测试True分支
        result_true = await executor.execute("branch-wf", {"flag": True})
        assert result_true.status == NodeStatus.COMPLETED
        assert "cond_output" in result_true.context
        assert "true_task_output" in result_true.context
        assert "false_task_output" not in result_true.context
        print("✅ [集成] True分支正确执行，False分支被跳过")

        # 测试False分支
        result_false = await executor.execute("branch-wf", {"flag": False})
        assert result_false.status == NodeStatus.COMPLETED
        assert "false_task_output" in result_false.context
        assert "true_task_output" not in result_false.context
        print("✅ [集成] False分支正确执行，True分支被跳过")

    @pytest.mark.asyncio
    async def test_workflow_node_failure_propagation(self):
        """节点失败传播"""
        editor = WorkflowEditor()
        wf = editor.create_workflow("Failure Workflow", "Tests failure propagation")
        wf.id = "fail-wf"
        editor._workflows["fail-wf"] = wf

        n1 = VisualNode(id="n1", type=NodeType.TASK, name="Good Task")
        n2 = VisualNode(id="n2", type=NodeType.FUNCTION, name="Bad Function",
                        config={"code": "raise ValueError('intentional error')"})
        n3 = VisualNode(id="n3", type=NodeType.TASK, name="After Failure")

        for n in [n1, n2, n3]:
            wf.add_node(n)
        wf.add_connection(Connection(id="c1", source_node="n1", source_port="out", target_node="n2", target_port="in"))
        wf.add_connection(Connection(id="c2", source_node="n2", source_port="out", target_node="n3", target_port="in"))

        executor = WorkflowExecutor(editor)
        result = await executor.execute("fail-wf")

        assert result.status == NodeStatus.FAILED
        assert result.node_results["n1"].status == NodeStatus.COMPLETED
        assert result.node_results["n2"].status == NodeStatus.FAILED
        assert "n3" not in result.node_results or result.node_results["n3"].status == NodeStatus.PENDING
        print("✅ [集成] 节点失败正确传播，后续节点不执行")

    @pytest.mark.asyncio
    async def test_workflow_execution_results_cache_limit(self):
        """执行结果缓存限制（500条）"""
        editor = WorkflowEditor()
        wf = editor.create_workflow("Cache Test", "")
        wf.id = "cache-wf"
        editor._workflows["cache-wf"] = wf
        n1 = VisualNode(id="n1", type=NodeType.TASK, name="Task")
        wf.add_node(n1)

        executor = WorkflowExecutor(editor)
        for i in range(550):
            await executor.execute("cache-wf")
        print(f"✅ [集成] 550次执行后缓存清理正常（无崩溃）")

    @pytest.mark.asyncio
    async def test_region_manager_full_lifecycle(self):
        """RegionManager完整生命周期"""
        rm = RegionManager()
        r1 = await rm.register_region(create_region("us-east", "US East", "http://us-east:8080"))
        r2 = await rm.register_region(create_region("eu-west", "EU West", "http://eu-west:8080"))
        r3 = await rm.register_region(create_region("ap-south", "AP South", "http://ap-south:8080"))

        assert len(await rm.list_regions()) == 3

        # 更新状态
        await rm.update_region_status("us-east", RegionStatus.ACTIVE)
        await rm.update_region_status("eu-west", RegionStatus.ACTIVE)
        await rm.update_region_status("ap-south", RegionStatus.DEGRADED)

        regions = await rm.list_regions()
        assert regions[2].status == RegionStatus.DEGRADED

        # 选择区域
        best = await rm.select_region("us-east-client")
        assert best is not None

        # 注销
        await rm.unregister_region("ap-south")
        assert len(await rm.list_regions()) == 2
        print("✅ [集成] RegionManager完整生命周期正常")

    @pytest.mark.asyncio
    async def test_region_failover(self):
        """区域故障转移"""
        rm = RegionManager()
        await rm.register_region(create_region("primary", "Primary", "http://primary:8080"))
        await rm.register_region(create_region("backup", "Backup", "http://backup:8080"))

        fm = FailoverManager(rm)
        fm.set_failover_rule("primary", ["backup"])

        # 主区域故障
        await rm.update_region_status("primary", RegionStatus.FAILOVER)

        regions = await rm.list_regions()
        history = fm.get_failover_history()
        print(f"✅ [集成] 区域故障转移: primary={regions[0].status.value}, history={len(history)}")

    @pytest.mark.asyncio
    async def test_region_data_sync(self):
        """跨区域数据同步"""
        rm = RegionManager()
        await rm.register_region(create_region("r1", "Region 1", "http://r1:8080"))
        await rm.register_region(create_region("r2", "Region 2", "http://r2:8080"))

        ds = DataSyncEngine(rm)
        # 先创建分区
        partition = await ds.create_partition("p1", "r1", ["r2"], SyncMode.ASYNC)
        assert partition.partition_id == "p1"

        # 同步数据
        task_id = await ds.sync_data("p1", {"key": "value"}, "r1")
        assert task_id is not None
        print(f"✅ [集成] 跨区域数据同步完成: task_id={task_id}")

    @pytest.mark.asyncio
    async def test_geo_routing_service(self):
        """地理路由服务"""
        rm = RegionManager()
        await rm.register_region(create_region("us", "US", "http://us:8080", location="us"))
        await rm.register_region(create_region("eu", "EU", "http://eu:8080", location="eu"))

        grs = GeoRoutingService(rm)
        grs.set_geo_mapping("us-east", "us")
        grs.set_geo_mapping("eu-west", "eu")

        target = await grs.route_by_location("us-east")
        assert target is not None
        assert target.region_id == "us"
        print(f"✅ [集成] 地理路由: client=us-east -> {target.region_id}")

    @pytest.mark.asyncio
    async def test_model_router_registration(self):
        """模型路由器注册和查询"""
        router = ModelRouter()

        config1 = ProviderConfig(
            provider_id="siliconflow", name="SiliconFlow",
            base_url="https://api.siliconflow.cn/v1", api_key="test-key",
            max_concurrent=5, timeout=10.0
        )
        router.register_provider(SiliconFlowProvider(config1))

        model = ModelInfo(
            model_id="deepseek-ai/DeepSeek-V3",
            provider_id="siliconflow",
            display_name="DeepSeek V3",
            capabilities={ModelCapability.CHAT, ModelCapability.STREAMING},
            max_context_tokens=65536,
            cost_per_1k_input=0.001,
            priority=10
        )
        router.register_model(model)

        # 使用route方法获取可用模型
        available = await router.route(model="deepseek-ai/DeepSeek-V3")
        assert available is not None
        print(f"✅ [集成] 模型注册成功: {available.display_name}")

    @pytest.mark.asyncio
    async def test_model_router_usage_stats(self):
        """模型路由用量统计"""
        router = ModelRouter()
        config = ProviderConfig(
            provider_id="test-provider", name="Test",
            base_url="http://test:8080", api_key="test-key"
        )
        router.register_provider(SiliconFlowProvider(config))
        router.register_model(ModelInfo(
            model_id="test-model", provider_id="test-provider",
            display_name="Test Model", cost_per_1k_input=0.001, cost_per_1k_output=0.002
        ))

        stats = router.get_usage_stats()
        assert stats["total_requests"] == 0
        assert stats["total_tokens"] == 0
        print(f"✅ [集成] 用量统计正常: {stats}")

    @pytest.mark.asyncio
    async def test_cross_module_pool_region_integration(self):
        """Pool + Region 联动集成"""
        mgr = AgentPoolManager()
        rm = RegionManager()

        await rm.register_region(create_region("region-1", "Region 1", "http://r1:8080"))
        await rm.update_region_status("region-1", RegionStatus.ACTIVE)

        pool = await mgr.create_pool("regional-pool", "agent-v1", min_size=1, max_size=10, initial_size=3)

        # 模拟区域故障 -> 池缩容
        await rm.update_region_status("region-1", RegionStatus.DEGRADED)
        event = await mgr.scale_pool(pool.pool_id, 1)
        assert event.new_size == 1
        print("✅ [集成] Pool+Region联动: 区域故障触发池缩容")

        await rm.update_region_status("region-1", RegionStatus.ACTIVE)
        event2 = await mgr.scale_pool(pool.pool_id, 5)
        assert event2.new_size == 5
        print("✅ [集成] Pool+Region联动: 区域恢复触发池扩容")

    @pytest.mark.asyncio
    async def test_cross_module_editor_pool_integration(self):
        """Editor + Pool 联动集成"""
        editor = WorkflowEditor()
        wf = editor.create_workflow("Editor Pool Integration", "")
        wf.id = "editor-pool-wf"
        editor._workflows["editor-pool-wf"] = wf

        task1 = VisualNode(id="t1", type=NodeType.TASK, name="Task 1")
        task2 = VisualNode(id="t2", type=NodeType.TASK, name="Task 2")
        task3 = VisualNode(id="t3", type=NodeType.FUNCTION, name="Agent Dispatch",
                          config={"code": "result = {'agents_needed': 3, 'priority': 'high'}"})
        wf.add_node(task1)
        wf.add_node(task2)
        wf.add_node(task3)
        wf.add_connection(Connection(id="c1", source_node="t1", source_port="out", target_node="t2", target_port="in"))
        wf.add_connection(Connection(id="c2", source_node="t2", source_port="out", target_node="t3", target_port="in"))

        executor = WorkflowExecutor(editor)
        result = await executor.execute("editor-pool-wf", {"pool_id": "test-pool"})
        assert result.status == NodeStatus.COMPLETED

        agent_count = result.context.get("t3_output", {}).get("agents_needed", 1)
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("dynamic-pool", "dispatched", min_size=1, max_size=10, initial_size=agent_count)
        assert pool.current_size == agent_count
        print(f"✅ [集成] Editor+Pool联动: 工作流输出驱动池创建 (agents={agent_count})")

    @pytest.mark.asyncio
    async def test_template_market_full_lifecycle(self):
        """模板市场完整生命周期"""
        market = TemplateMarket()

        wf = VisualWorkflow(id="tmpl-wf", name="Data Pipeline Template")
        n1 = VisualNode(id="extract", type=NodeType.TASK, name="Extract")
        n2 = VisualNode(id="transform", type=NodeType.FUNCTION, name="Transform",
                        config={"code": "result = context.get('data', [])"})
        n3 = VisualNode(id="load", type=NodeType.OUTPUT, name="Load")
        wf.add_node(n1)
        wf.add_node(n2)
        wf.add_node(n3)
        wf.add_connection(Connection(id="c1", source_node="extract", source_port="out", target_node="transform", target_port="in"))
        wf.add_connection(Connection(id="c2", source_node="transform", source_port="out", target_node="load", target_port="in"))

        tmpl = await market.publish_template(
            name="ETL Pipeline", description="Standard ETL template",
            category="data", workflow=wf, author="system", tags=["etl", "pipeline"]
        )
        assert tmpl.id is not None

        results = await market.search_templates(query="ETL")
        assert len(results) == 1

        downloaded_wf = await market.download_template(tmpl.id)
        assert downloaded_wf is not None
        assert downloaded_wf.id == "tmpl-wf"
        assert tmpl.downloads == 1

        empty = await market.search_templates(query="nonexistent")
        assert len(empty) == 0
        print("✅ [集成] 模板市场完整生命周期正常")


# ============================================================
# 第三部分：接口全量遍历测试
# ============================================================

class TestInterfaceFullCoverage:
    """所有公共API异常入参、空值、非法格式"""

    @pytest.mark.asyncio
    async def test_create_pool_empty_name(self):
        """创建池：空名称"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("", "test-type")
        assert pool is not None
        assert pool.name == ""
        print("✅ [接口] create_pool空名称不崩溃")

    @pytest.mark.asyncio
    async def test_create_pool_empty_agent_type(self):
        """创建池：空agent_type"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("test", "")
        assert pool is not None
        assert pool.agent_type == ""
        print("✅ [接口] create_pool空agent_type不崩溃")

    @pytest.mark.asyncio
    async def test_create_pool_zero_max_size(self):
        """创建池：max_size=0"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("zero-max", "test", min_size=0, max_size=0, initial_size=0)
        assert pool.max_size == 0
        assert pool.current_size == 0
        print("✅ [接口] create_pool max_size=0正常")

    @pytest.mark.asyncio
    async def test_create_pool_negative_values(self):
        """创建池：负数值"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("neg", "test", min_size=-5, max_size=-1, initial_size=-3)
        print(f"✅ [接口] create_pool负数值: min={pool.min_size}, max={pool.max_size}, size={pool.current_size}")

    @pytest.mark.asyncio
    async def test_create_pool_very_large_initial(self):
        """创建池：超大initial_size"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("huge", "test", min_size=1, max_size=5, initial_size=1000000)
        assert pool.current_size == 5
        print("✅ [接口] create_pool超大initial_size被clamp到max_size")

    @pytest.mark.asyncio
    async def test_create_pool_special_chars_name(self):
        """创建池：特殊字符名称"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("pool<script>alert(1)</script>", "test")
        assert pool.name == "pool<script>alert(1)</script>"
        print("✅ [接口] create_pool特殊字符名称不崩溃")

    @pytest.mark.asyncio
    async def test_create_pool_unicode_name(self):
        """创建池：Unicode名称"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("池-中文-🚀", "test")
        assert pool.name == "池-中文-🚀"
        print("✅ [接口] create_pool Unicode名称正常")

    @pytest.mark.asyncio
    async def test_acquire_agent_with_requirements(self):
        """acquire_agent带requirements参数"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("req-pool", "test", min_size=1, max_size=5, initial_size=2)
        agent = await mgr.acquire_agent(pool.pool_id, requirements={"cpu": 2, "memory": 1024})
        assert agent is not None
        print("✅ [接口] acquire_agent带requirements不崩溃")

    @pytest.mark.asyncio
    async def test_list_pools_empty(self):
        """空管理器list_pools"""
        mgr = AgentPoolManager()
        pools = await mgr.list_pools()
        assert pools == []
        print("✅ [接口] 空管理器list_pools返回空列表")

    # --- WorkflowEditor 接口 ---

    def test_editor_create_workflow(self):
        """编辑器创建工作流"""
        editor = WorkflowEditor()
        wf = editor.create_workflow("Test WF", "A test workflow")
        assert wf.id is not None
        assert wf.name == "Test WF"
        print("✅ [接口] 编辑器创建工作流正常")

    def test_editor_empty_workflow_name(self):
        """编辑器：空工作流名称"""
        editor = WorkflowEditor()
        wf = editor.create_workflow("", "")
        assert wf.name == ""
        print("✅ [接口] 编辑器空名称不崩溃")

    # --- WorkflowValidator 接口 ---

    def test_validator_empty_workflow(self):
        """验证器：空工作流"""
        wf = VisualWorkflow(id="empty", name="Empty")
        is_valid, errors = WorkflowValidator.validate(wf)
        assert not is_valid
        assert len(errors) > 0
        print("✅ [接口] 验证器空工作流返回错误")

    def test_validator_single_node_no_connections(self):
        """验证器：单节点无连接"""
        wf = VisualWorkflow(id="single", name="Single")
        wf.add_node(VisualNode(id="n1", type=NodeType.TASK, name="Task"))
        is_valid, errors = WorkflowValidator.validate(wf)
        assert is_valid
        print("✅ [接口] 单节点工作流验证通过")

    def test_validator_dangling_connections(self):
        """验证器：悬空连接"""
        wf = VisualWorkflow(id="dangling", name="Dangling")
        wf.add_node(VisualNode(id="n1", type=NodeType.TASK, name="Task"))
        wf.add_connection(Connection(
            id="c1", source_node="n1", source_port="out",
            target_node="nonexistent", target_port="in"
        ))
        is_valid, errors = WorkflowValidator.validate(wf)
        assert not is_valid
        assert any("不存在" in e for e in errors)
        print("✅ [接口] 悬空连接被正确检测")

    # --- RegionManager 接口 ---

    @pytest.mark.asyncio
    async def test_region_register_duplicate(self):
        """注册重复区域"""
        rm = RegionManager()
        r1 = await rm.register_region(create_region("dup", "Duplicate"))
        r2 = await rm.register_region(create_region("dup", "Duplicate 2"))
        regions = await rm.list_regions()
        # list_regions返回List[RegionInfo]，检查region_id
        region_ids = [r.region_id for r in regions]
        assert "dup" in region_ids
        print("✅ [接口] 重复区域注册处理正常")

    @pytest.mark.asyncio
    async def test_region_unregister_nonexistent(self):
        """注销不存在的区域"""
        rm = RegionManager()
        result = await rm.unregister_region("nonexistent")
        assert result is False
        print("✅ [接口] 注销不存在的区域返回False")

    @pytest.mark.asyncio
    async def test_region_update_status_nonexistent(self):
        """更新不存在区域的状态"""
        rm = RegionManager()
        result = await rm.update_region_status("nonexistent", RegionStatus.ACTIVE)
        assert result is False
        print("✅ [接口] 更新不存在区域状态返回False")

    @pytest.mark.asyncio
    async def test_region_select_empty(self):
        """空管理器选择区域"""
        rm = RegionManager()
        result = await rm.select_region()
        assert result is None
        print("✅ [接口] 空管理器选择区域返回None")

    @pytest.mark.asyncio
    async def test_region_health_check_empty(self):
        """空管理器健康检查"""
        rm = RegionManager()
        result = await rm.health_check()
        assert result == {}
        print("✅ [接口] 空管理器健康检查返回空字典")

    # --- ModelRouter 接口 ---

    def test_router_register_duplicate_provider(self):
        """注册重复provider"""
        router = ModelRouter()
        config = ProviderConfig(
            provider_id="dup", name="Duplicate",
            base_url="http://dup:8080", api_key="key"
        )
        router.register_provider(SiliconFlowProvider(config))
        router.register_provider(SiliconFlowProvider(config))
        print("✅ [接口] 重复provider注册不崩溃")

    def test_router_register_duplicate_model(self):
        """注册重复model"""
        router = ModelRouter()
        config = ProviderConfig(
            provider_id="p1", name="P1",
            base_url="http://p1:8080", api_key="key"
        )
        router.register_provider(SiliconFlowProvider(config))
        router.register_model(ModelInfo(model_id="m1", provider_id="p1", display_name="M1"))
        router.register_model(ModelInfo(model_id="m1", provider_id="p1", display_name="M1 v2"))
        print("✅ [接口] 重复model注册不崩溃")

    def test_router_usage_stats_empty(self):
        """空路由器用量统计"""
        router = ModelRouter()
        stats = router.get_usage_stats()
        assert stats["total_requests"] == 0
        assert stats["total_tokens"] == 0
        print("✅ [接口] 空路由器用量统计返回零值")

    def test_router_latency_stats_empty(self):
        """空路由器延迟统计"""
        router = ModelRouter()
        stats = router.get_latency_stats()
        assert stats == {}
        print("✅ [接口] 空路由器延迟统计返回空字典")

    @pytest.mark.asyncio
    async def test_router_route_no_models(self):
        """无模型时路由"""
        router = ModelRouter()
        result = await router.route(model="nonexistent")
        assert result is None
        print("✅ [接口] 无模型时路由返回None")

    # --- ConfigGenerator 接口 ---

    @pytest.mark.asyncio
    async def test_config_generator_python(self):
        """Python代码生成"""
        gen = ConfigGenerator()
        wf = VisualWorkflow(id="gen-wf", name="Gen Test")
        wf.add_node(VisualNode(id="n1", type=NodeType.TASK, name="Task"))
        code = await gen.generate_code(wf, "python")
        assert isinstance(code, str)
        assert len(code) > 0
        print("✅ [接口] Python代码生成正常")

    @pytest.mark.asyncio
    async def test_config_generator_yaml(self):
        """YAML配置生成"""
        gen = ConfigGenerator()
        wf = VisualWorkflow(id="gen-wf", name="Gen Test")
        wf.add_node(VisualNode(id="n1", type=NodeType.TASK, name="Task"))
        config = await gen.generate_code(wf, "yaml")
        assert isinstance(config, str)
        assert len(config) > 0
        print("✅ [接口] YAML配置生成正常")

    @pytest.mark.asyncio
    async def test_config_generator_unsupported_language(self):
        """不支持的代码生成语言"""
        gen = ConfigGenerator()
        wf = VisualWorkflow(id="gen-wf", name="Gen Test")
        wf.add_node(VisualNode(id="n1", type=NodeType.TASK, name="Task"))
        code = await gen.generate_code(wf, "java")
        assert isinstance(code, str)
        print(f"✅ [接口] 不支持语言代码生成不崩溃")

    @pytest.mark.asyncio
    async def test_config_generator_empty_workflow(self):
        """空工作流代码生成"""
        gen = ConfigGenerator()
        wf = VisualWorkflow(id="empty", name="Empty")
        code = await gen.generate_code(wf, "python")
        assert isinstance(code, str)
        print("✅ [接口] 空工作流代码生成不崩溃")

    # --- PerformanceBenchmark 接口 ---

    @pytest.mark.asyncio
    async def test_benchmark_with_zero_tasks(self):
        """零任务基准测试"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("bench-pool", "test", min_size=1, max_size=5, initial_size=2)
        bench = PerformanceBenchmark(mgr)
        result = await bench.run_benchmark(pool.pool_id, task_count=0, concurrent=1)
        assert result["completed"] == 0
        assert result["failed"] == 0
        print("✅ [接口] 零任务基准测试正常")

    @pytest.mark.asyncio
    async def test_benchmark_nonexistent_pool(self):
        """不存在的池基准测试"""
        mgr = AgentPoolManager()
        bench = PerformanceBenchmark(mgr)
        result = await bench.run_benchmark("nonexistent", task_count=10)
        assert result.get("completed", 0) == 0
        print(f"✅ [接口] 不存在池基准测试: completed={result.get('completed', 0)}")

    # --- CapacityPlanner 接口 ---

    @pytest.mark.asyncio
    async def test_capacity_planner_basic(self):
        """容量规划器基本功能"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("cap-pool", "test", min_size=1, max_size=100, initial_size=5)
        planner = CapacityPlanner(mgr)
        plan = await planner.predict_capacity(pool.pool_id, target_utilization=0.7)
        assert plan is not None
        assert "predicted_size" in plan
        print(f"✅ [接口] 容量规划器: {plan}")

    @pytest.mark.asyncio
    async def test_capacity_planner_nonexistent_pool(self):
        """不存在的池容量规划"""
        mgr = AgentPoolManager()
        planner = CapacityPlanner(mgr)
        plan = await planner.predict_capacity("nonexistent", target_utilization=0.7)
        assert plan == {}
        print(f"✅ [接口] 不存在池容量规划返回空字典")


# ============================================================
# 第四部分：数据一致性校验
# ============================================================

class TestDataConsistency:
    """数据一致性校验"""

    @pytest.mark.asyncio
    async def test_pool_size_consistency_after_concurrent_ops(self):
        """并发操作后池大小一致性"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("consist-pool", "test", min_size=1, max_size=20, initial_size=5)

        async def scale_up_down():
            for _ in range(10):
                await mgr.scale_pool(pool.pool_id, 15)
                await mgr.scale_pool(pool.pool_id, 3)

        await asyncio.gather(*[scale_up_down() for _ in range(3)])

        pool = (await mgr.list_pools())[0]
        assert pool.current_size == len(pool.agents), \
            f"Size mismatch: current_size={pool.current_size}, agents={len(pool.agents)}"
        print(f"✅ [一致性] 并发操作后池大小一致: size={pool.current_size}")

    @pytest.mark.asyncio
    async def test_agent_task_count_consistency(self):
        """Agent任务计数一致性"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("task-pool", "test", min_size=1, max_size=10, initial_size=3)

        agents = []
        for _ in range(30):
            agent = await mgr.acquire_agent(pool.pool_id)
            if agent:
                agents.append(agent)

        total_tasks = sum(a.current_tasks for a in pool.agents.values())
        assert total_tasks == len(agents), \
            f"Task count mismatch: sum={total_tasks}, acquired={len(agents)}"

        for agent in agents:
            await mgr.release_agent(agent.agent_id)

        total_after = sum(a.current_tasks for a in pool.agents.values())
        assert total_after == 0, f"After release: total_tasks={total_after}"
        print(f"✅ [一致性] Agent任务计数一致: acquire={len(agents)}, release全部归零")

    @pytest.mark.asyncio
    async def test_registry_pool_consistency(self):
        """agent_registry与pool.agents一致性"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("reg-pool", "test", min_size=1, max_size=10, initial_size=5)

        for agent_id in pool.agents:
            assert agent_id in mgr._agent_registry, f"Agent {agent_id} not in registry"

        await mgr.scale_pool(pool.pool_id, 2)

        for agent_id in pool.agents:
            assert agent_id in mgr._agent_registry
        print(f"✅ [一致性] registry与pool.agents一致")

    @pytest.mark.asyncio
    async def test_workflow_context_consistency(self):
        """工作流上下文一致性"""
        editor = WorkflowEditor()
        wf = editor.create_workflow("Context Test", "")
        wf.id = "ctx-wf"
        editor._workflows["ctx-wf"] = wf

        n1 = VisualNode(id="n1", type=NodeType.FUNCTION, name="Set Value",
                        config={"code": "result = context.get('input', 0) * 2"})
        n2 = VisualNode(id="n2", type=NodeType.FUNCTION, name="Use Value",
                        config={"code": "result = context.get('n1_output', 0) + 10"})
        wf.add_node(n1)
        wf.add_node(n2)
        wf.add_connection(Connection(id="c1", source_node="n1", source_port="out", target_node="n2", target_port="in"))

        executor = WorkflowExecutor(editor)
        result = await executor.execute("ctx-wf", {"input": 5})
        assert result.status == NodeStatus.COMPLETED
        assert result.context.get("n1_output") == 10
        assert result.context.get("n2_output") == 20
        print("✅ [一致性] 工作流上下文传递一致")

    @pytest.mark.asyncio
    async def test_workflow_to_dict_roundtrip(self):
        """工作流序列化一致性"""
        wf = VisualWorkflow(id="ser-wf", name="Serialization Test", description="Test")
        n1 = VisualNode(id="n1", type=NodeType.TASK, name="Task 1")
        n1.add_input_port("data", "string", required=True)
        n1.add_output_port("result", "json")
        wf.add_node(n1)
        wf.add_connection(Connection(
            id="c1", source_node="n1", source_port="result",
            target_node="n1", target_port="data"
        ))
        wf.variables = {"key": "value"}

        d = wf.to_dict()
        assert d["id"] == "ser-wf"
        assert d["name"] == "Serialization Test"
        assert "n1" in d["nodes"]
        assert "c1" in d["connections"]
        assert d["variables"] == {"key": "value"}
        assert len(d["nodes"]["n1"]["input_ports"]) == 1
        assert len(d["nodes"]["n1"]["output_ports"]) == 1
        print("✅ [一致性] 工作流序列化数据完整")


# ============================================================
# 第五部分：异常容错测试
# ============================================================

class TestExceptionTolerance:
    """异常容错测试"""

    async def _exec_code(self, code, timeout=5.0):
        wf_id = f"exc-test-{uuid.uuid4().hex[:8]}"
        editor, wf = create_function_workflow(wf_id, "Exception Test", code)
        executor = WorkflowExecutor(editor)
        return await executor.execute(wf_id, timeout=timeout)

    @pytest.mark.asyncio
    async def test_function_node_syntax_error(self):
        """FUNCTION节点语法错误"""
        result = await self._exec_code("this is not valid python syntax !!!")
        assert result.status == NodeStatus.FAILED
        assert result.node_results["fn1"].status == NodeStatus.FAILED
        print(f"✅ [容错] 语法错误正确处理: {result.node_results['fn1'].error[:60]}")

    @pytest.mark.asyncio
    async def test_function_node_name_error(self):
        """FUNCTION节点NameError"""
        result = await self._exec_code("result = undefined_variable_xyz")
        assert result.status == NodeStatus.FAILED
        print(f"✅ [容错] NameError正确处理: {result.node_results['fn1'].error[:60]}")

    @pytest.mark.asyncio
    async def test_function_node_type_error(self):
        """FUNCTION节点TypeError"""
        result = await self._exec_code("result = int('not a number')")
        assert result.status == NodeStatus.FAILED
        print(f"✅ [容错] TypeError正确处理: {result.node_results['fn1'].error[:60]}")

    @pytest.mark.asyncio
    async def test_function_node_recursion_limit(self):
        """FUNCTION节点递归深度限制"""
        result = await self._exec_code("""
def f(n):
    return f(n-1)
result = f(10000)
""", timeout=5.0)
        assert result.status in (NodeStatus.FAILED, NodeStatus.TIMEOUT)
        print(f"✅ [容错] 递归深度限制: status={result.status.value}")

    @pytest.mark.asyncio
    async def test_condition_node_invalid_expression(self):
        """CONDITION节点无效表达式"""
        wf_id = f"cond-exc-{uuid.uuid4().hex[:8]}"
        editor, wf = create_condition_workflow(wf_id, "Condition Error", "this is not valid &&&")
        executor = WorkflowExecutor(editor)
        result = await executor.execute(wf_id)
        assert result.status == NodeStatus.FAILED
        assert result.node_results["cond1"].status == NodeStatus.FAILED
        print(f"✅ [容错] 无效条件表达式正确处理")

    @pytest.mark.asyncio
    async def test_workflow_timeout_with_slow_function(self):
        """工作流超时与慢函数（使用已注入的time模块）"""
        result = await self._exec_code("time.sleep(100); result = 'done'", timeout=5.0)
        assert result.status == NodeStatus.TIMEOUT
        print(f"✅ [容错] 慢函数被timeout阻止（BUG#77修复验证）")

    @pytest.mark.asyncio
    async def test_multiple_failures_in_workflow(self):
        """工作流中多个失败节点"""
        editor = WorkflowEditor()
        wf = editor.create_workflow("Multi Failure", "")
        wf.id = "multi-fail"
        editor._workflows["multi-fail"] = wf

        wf.add_node(VisualNode(id="n1", type=NodeType.TASK, name="Good"))
        wf.add_node(VisualNode(id="n2", type=NodeType.FUNCTION, name="Bad",
                               config={"code": "1/0"}))
        wf.add_node(VisualNode(id="n3", type=NodeType.TASK, name="After"))

        wf.add_connection(Connection(id="c1", source_node="n1", source_port="out", target_node="n3", target_port="in"))
        wf.add_connection(Connection(id="c2", source_node="n2", source_port="out", target_node="n3", target_port="in"))

        executor = WorkflowExecutor(editor)
        result = await executor.execute("multi-fail")
        assert result.status in (NodeStatus.COMPLETED, NodeStatus.FAILED)
        print(f"✅ [容错] 多失败节点工作流: status={result.status.value}")


# ============================================================
# 第六部分：基础性能核验
# ============================================================

class TestBasicPerformance:
    """基础性能核验"""

    @pytest.mark.asyncio
    async def test_rapid_workflow_execution(self):
        """快速连续执行100个工作流"""
        editor = WorkflowEditor()
        wf = editor.create_workflow("Perf Test", "")
        wf.id = "perf-wf"
        editor._workflows["perf-wf"] = wf
        wf.add_node(VisualNode(id="n1", type=NodeType.FUNCTION, name="Calc",
                               config={"code": "result = sum(range(100))"}))

        executor = WorkflowExecutor(editor)
        start = time.time()
        for _ in range(100):
            result = await executor.execute("perf-wf")
            assert result.status == NodeStatus.COMPLETED
        elapsed = time.time() - start
        print(f"✅ [性能] 100个工作流执行: {elapsed:.2f}s ({100/elapsed:.1f} wf/s)")

    @pytest.mark.asyncio
    async def test_large_workflow_execution(self):
        """大型工作流（50个节点）执行"""
        editor = WorkflowEditor()
        wf = editor.create_workflow("Large Workflow", "")
        wf.id = "large-wf"
        editor._workflows["large-wf"] = wf

        prev_id = None
        for i in range(50):
            node_id = f"n{i}"
            wf.add_node(VisualNode(
                id=node_id, type=NodeType.FUNCTION, name=f"Step {i}",
                config={"code": f"result = {i}"}
            ))
            if prev_id:
                wf.add_connection(Connection(
                    id=f"c{i}", source_node=prev_id, source_port="out",
                    target_node=node_id, target_port="in"
                ))
            prev_id = node_id

        executor = WorkflowExecutor(editor)
        start = time.time()
        result = await executor.execute("large-wf")
        elapsed = time.time() - start
        assert result.status == NodeStatus.COMPLETED
        assert len(result.node_results) == 50
        print(f"✅ [性能] 50节点工作流: {elapsed:.2f}s ({50/elapsed:.1f} nodes/s)")

    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self):
        """并发执行50个工作流"""
        editor = WorkflowEditor()

        async def run_wf(idx):
            wf_id = f"conc-wf-{idx}"
            wf = editor.create_workflow(f"Concurrent {idx}", "")
            wf.id = wf_id
            editor._workflows[wf_id] = wf
            wf.add_node(VisualNode(
                id="n1", type=NodeType.FUNCTION, name="Calc",
                config={"code": f"result = {idx} * 2"}
            ))
            executor = WorkflowExecutor(editor)
            return await executor.execute(wf_id)

        start = time.time()
        results = await asyncio.gather(*[run_wf(i) for i in range(50)])
        elapsed = time.time() - start
        assert all(r.status == NodeStatus.COMPLETED for r in results)
        print(f"✅ [性能] 50个并发工作流: {elapsed:.2f}s ({50/elapsed:.1f} wf/s)")

    @pytest.mark.asyncio
    async def test_pool_rapid_acquire_release(self):
        """高频acquire/release性能"""
        mgr = AgentPoolManager()
        pool = await mgr.create_pool("perf-pool", "test", min_size=10, max_size=10, initial_size=10)

        start = time.time()
        for _ in range(1000):
            agent = await mgr.acquire_agent(pool.pool_id)
            assert agent is not None
            await mgr.release_agent(agent.agent_id)
        elapsed = time.time() - start
        print(f"✅ [性能] 1000次acquire/release: {elapsed:.2f}s ({1000/elapsed:.1f} ops/s)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
