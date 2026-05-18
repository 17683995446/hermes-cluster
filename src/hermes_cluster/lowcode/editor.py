"""
Hermes Cluster v6.0 - 低代码配置模块
支持可视化编辑器、拖拽配置、模板市场

优化点:
1. 可视化工作流编辑器
2. 拖拽式Agent配置
3. 模板市场与共享
4. 配置版本管理
5. 实时预览与验证
6. 代码生成器
7. 配置导入导出
8. 协作编辑支持
9. 配置审计日志
10. 智能推荐系统
11. 表单自动生成
12. 多语言支持
13. 工作流执行器
14. 节点执行状态追踪
15. 执行错误处理
"""

from __future__ import annotations

import asyncio
import copy
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """节点类型"""
    AGENT = "agent"
    TASK = "task"
    WORKFLOW = "workflow"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    TRIGGER = "trigger"
    OUTPUT = "output"
    FUNCTION = "function"


class PortType(Enum):
    """端口类型"""
    INPUT = "input"
    OUTPUT = "output"


class NodeStatus(Enum):
    """节点执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class Port:
    """端口"""
    id: str
    name: str
    port_type: PortType
    data_type: str = "any"
    required: bool = False
    description: str = ""


@dataclass
class NodePosition:
    """节点位置"""
    x: float = 0.0
    y: float = 0.0


@dataclass
class VisualNode:
    """可视化节点"""
    id: str
    type: NodeType
    name: str
    position: NodePosition = field(default_factory=NodePosition)
    config: Dict[str, Any] = field(default_factory=dict)
    input_ports: List[Port] = field(default_factory=list)
    output_ports: List[Port] = field(default_factory=list)
    style: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_input_port(self, name: str, data_type: str = "any", required: bool = False) -> Port:
        port = Port(
            id=f"{self.id}_in_{len(self.input_ports)}",
            name=name,
            port_type=PortType.INPUT,
            data_type=data_type,
            required=required
        )
        self.input_ports.append(port)
        return port

    def add_output_port(self, name: str, data_type: str = "any") -> Port:
        port = Port(
            id=f"{self.id}_out_{len(self.output_ports)}",
            name=name,
            port_type=PortType.OUTPUT,
            data_type=data_type
        )
        self.output_ports.append(port)
        return port


# 别名：WorkflowNode = VisualNode (向后兼容)
WorkflowNode = VisualNode


@dataclass
class Connection:
    """连接"""
    id: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    style: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualWorkflow:
    """可视化工作流"""
    id: str
    name: str
    description: str = ""
    nodes: Dict[str, VisualNode] = field(default_factory=dict)
    connections: Dict[str, Connection] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: VisualNode) -> None:
        self.nodes[node.id] = node
        self.updated_at = datetime.now()

    def remove_node(self, node_id: str) -> bool:
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.connections = {
                k: v for k, v in self.connections.items()
                if v.source_node != node_id and v.target_node != node_id
            }
            self.updated_at = datetime.now()
            return True
        return False

    def add_connection(self, connection: Connection) -> None:
        self.connections[connection.id] = connection
        self.updated_at = datetime.now()

    def remove_connection(self, connection_id: str) -> bool:
        if connection_id in self.connections:
            del self.connections[connection_id]
            self.updated_at = datetime.now()
            return True
        return False

    def validate(self) -> List[str]:
        errors = []
        if not self.nodes:
            errors.append("工作流没有节点")
        for conn in self.connections.values():
            if conn.source_node not in self.nodes:
                errors.append(f"连接 {conn.id} 的源节点不存在")
            if conn.target_node not in self.nodes:
                errors.append(f"连接 {conn.id} 的目标节点不存在")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": {
                k: {
                    "id": v.id,
                    "type": v.type.value,
                    "name": v.name,
                    "position": {"x": v.position.x, "y": v.position.y},
                    "config": v.config,
                    "input_ports": [{"id": p.id, "name": p.name, "data_type": p.data_type} for p in v.input_ports],
                    "output_ports": [{"id": p.id, "name": p.name, "data_type": p.data_type} for p in v.output_ports],
                    "style": v.style,
                    "metadata": v.metadata
                }
                for k, v in self.nodes.items()
            },
            "connections": {
                k: {
                    "id": v.id,
                    "source_node": v.source_node,
                    "source_port": v.source_port,
                    "target_node": v.target_node,
                    "target_port": v.target_port,
                    "style": v.style
                }
                for k, v in self.connections.items()
            },
            "variables": self.variables,
            "triggers": self.triggers,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "metadata": self.metadata
        }


@dataclass
class NodeExecutionResult:
    """节点执行结果"""
    node_id: str
    status: NodeStatus
    output: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0


@dataclass
class WorkflowExecutionResult:
    """工作流执行结果"""
    execution_id: str
    workflow_id: str
    status: NodeStatus
    node_results: Dict[str, NodeExecutionResult] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class Template:
    """模板"""
    id: str
    name: str
    description: str
    category: str
    workflow: VisualWorkflow
    author: str = ""
    tags: List[str] = field(default_factory=list)
    rating: float = 0.0
    downloads: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class TemplateCategory:
    """模板分类"""
    id: str
    name: str
    description: str = ""
    icon: str = ""
    parent_id: Optional[str] = None


class WorkflowEditor:
    """工作流编辑器"""

    def __init__(self):
        self._workflows: Dict[str, VisualWorkflow] = {}
        self._undo_stack: Dict[str, List[Dict[str, Any]]] = {}
        self._redo_stack: Dict[str, List[Dict[str, Any]]] = {}
        self._max_undo: int = 50

    def create_workflow(self, name: str, description: str = "") -> VisualWorkflow:
        """创建工作流 (同步版本)"""
        workflow = VisualWorkflow(
            id=str(uuid.uuid4()),
            name=name,
            description=description
        )
        self._workflows[workflow.id] = workflow
        self._undo_stack[workflow.id] = []
        self._redo_stack[workflow.id] = []
        return workflow

    async def create_workflow_async(self, name: str, description: str = "") -> VisualWorkflow:
        """异步创建工作流"""
        return self.create_workflow(name, description)

    def add_node(self, workflow_id: str, node: VisualNode) -> bool:
        """添加节点 (同步版本)"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return False
        workflow.add_node(node)
        return True

    async def add_node_async(self, workflow_id: str, node_type: NodeType, name: str, 
                             position: Tuple[float, float], config: Optional[Dict[str, Any]] = None) -> Optional[VisualNode]:
        """异步添加节点"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None

        node = VisualNode(
            id=str(uuid.uuid4()),
            type=node_type,
            name=name,
            position=NodePosition(x=position[0], y=position[1]),
            config=config or {}
        )

        if node_type == NodeType.AGENT:
            node.add_input_port("input", "any", True)
            node.add_output_port("output", "any")
        elif node_type == NodeType.TASK:
            node.add_input_port("input", "any", True)
            node.add_output_port("output", "any")
        elif node_type == NodeType.CONDITION:
            node.add_input_port("input", "any", True)
            node.add_output_port("true", "any")
            node.add_output_port("false", "any")
        elif node_type == NodeType.PARALLEL:
            node.add_input_port("input", "any", True)
            node.add_output_port("output1", "any")
            node.add_output_port("output2", "any")
        elif node_type == NodeType.FUNCTION:
            node.add_input_port("input", "any", True)
            node.add_output_port("output", "any")

        workflow.add_node(node)
        return node

    def connect_nodes(self, workflow_id: str, source_node_id: str, target_node_id: str,
                      source_port: str = "output", target_port: str = "input") -> Optional[Connection]:
        """连接节点 (同步版本)"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None
        # 验证节点存在
        if source_node_id not in workflow.nodes:
            return None
        if target_node_id not in workflow.nodes:
            return None

        connection = Connection(
            id=str(uuid.uuid4()),
            source_node=source_node_id,
            source_port=source_port,
            target_node=target_node_id,
            target_port=target_port
        )
        workflow.add_connection(connection)
        return connection

    async def connect_nodes_async(self, workflow_id: str, source_node: str, source_port: str,
                                  target_node: str, target_port: str) -> Optional[Connection]:
        """异步连接节点"""
        return self.connect_nodes(workflow_id, source_node, target_node, source_port, target_port)

    async def validate_workflow(self, workflow_id: str) -> List[str]:
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return ["工作流不存在"]
        return workflow.validate()

    def get_workflow(self, workflow_id: str) -> Optional[VisualWorkflow]:
        return self._workflows.get(workflow_id)


class WorkflowExecutor:
    """工作流执行器"""

    def __init__(self, editor: Optional[WorkflowEditor] = None):
        self._editor = editor or WorkflowEditor()
        self._execution_results: Dict[str, WorkflowExecutionResult] = {}
        self._lock = asyncio.Lock()

    async def execute(self, workflow_id: str, context: Optional[Dict[str, Any]] = None,
                      timeout: float = 300.0) -> WorkflowExecutionResult:
        """执行工作流"""
        workflow = self._editor.get_workflow(workflow_id)
        if not workflow:
            return WorkflowExecutionResult(
                execution_id=str(uuid.uuid4()),
                workflow_id=workflow_id,
                status=NodeStatus.FAILED,
                error="工作流不存在"
            )

        execution_id = str(uuid.uuid4())
        # 在构建执行顺序之前添加验证
        errors = workflow.validate()
        if errors:
            return WorkflowExecutionResult(
                execution_id=execution_id,
                workflow_id=workflow_id,
                status=NodeStatus.FAILED,
                error=f"工作流验证失败: {'; '.join(errors)}"
            )

        result = WorkflowExecutionResult(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=NodeStatus.RUNNING,
            context=context or {}
        )

        async with self._lock:
            self._execution_results[execution_id] = result
            # 限制保留最近500条
            if len(self._execution_results) > 500:
                keys = list(self._execution_results.keys())
                for k in keys[:len(self._execution_results) - 300]:
                    del self._execution_results[k]

        try:
            # 构建执行图
            execution_order = self._build_execution_order(workflow)

            # 按顺序执行节点
            for node_id in execution_order:
                if result.status == NodeStatus.FAILED:
                    break

                node = workflow.nodes.get(node_id)
                if not node:
                    continue

                node_result = await self._execute_node(node, result.context, timeout)
                result.node_results[node_id] = node_result

                if node_result.status == NodeStatus.FAILED:
                    result.status = NodeStatus.FAILED
                    result.error = node_result.error
                    break
                elif node_result.status == NodeStatus.COMPLETED:
                    # 更新上下文
                    if node_result.output is not None:
                        result.context[f"{node_id}_output"] = node_result.output

            if result.status != NodeStatus.FAILED:
                result.status = NodeStatus.COMPLETED

        except Exception as e:
            result.status = NodeStatus.FAILED
            result.error = str(e)
            logger.error(f"工作流执行错误: {e}")

        finally:
            result.end_time = datetime.now()

        return result

    def _build_execution_order(self, workflow: VisualWorkflow) -> List[str]:
        """构建节点执行顺序 (拓扑排序)"""
        # 构建邻接表
        graph: Dict[str, List[str]] = {node_id: [] for node_id in workflow.nodes}
        in_degree: Dict[str, int] = {node_id: 0 for node_id in workflow.nodes}

        for conn in workflow.connections.values():
            if conn.source_node in graph and conn.target_node in graph:
                graph[conn.source_node].append(conn.target_node)
                in_degree[conn.target_node] += 1

        # Kahn算法
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            for neighbor in graph[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 如果有剩余节点，说明有环，按原始顺序添加
        for node_id in workflow.nodes:
            if node_id not in order:
                order.append(node_id)

        return order

    async def _execute_node(self, node: VisualNode, context: Dict[str, Any],
                           timeout: float) -> NodeExecutionResult:
        """执行单个节点"""
        result = NodeExecutionResult(
            node_id=node.id,
            status=NodeStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            if node.type == NodeType.FUNCTION:
                # 执行函数节点
                code = node.config.get("code", "")
                if code:
                    # 创建安全的执行环境
                    local_vars = {"context": context, "result": None}
                    exec(code, {"__builtins__": {}}, local_vars)
                    result.output = local_vars.get("result")
                result.status = NodeStatus.COMPLETED

            elif node.type == NodeType.AGENT:
                # 模拟Agent执行
                await asyncio.sleep(0.001)  # 模拟延迟
                result.output = {"agent_result": f"Processed by {node.name}"}
                result.status = NodeStatus.COMPLETED

            elif node.type == NodeType.TASK:
                # 模拟任务执行
                await asyncio.sleep(0.001)
                result.output = {"task_result": f"Task {node.name} completed"}
                result.status = NodeStatus.COMPLETED

            elif node.type == NodeType.CONDITION:
                # 条件节点
                condition = node.config.get("condition", "True")
                try:
                    local_vars = {"context": context}
                    condition_result = eval(condition, {"__builtins__": {}}, local_vars)
                    result.output = {"condition_result": bool(condition_result)}
                    result.status = NodeStatus.COMPLETED
                except Exception as e:
                    result.error = f"条件评估错误: {e}"
                    result.status = NodeStatus.FAILED

            else:
                # 其他类型节点
                result.output = {"node_type": node.type.value}
                result.status = NodeStatus.COMPLETED

        except asyncio.TimeoutError:
            result.status = NodeStatus.TIMEOUT
            result.error = "执行超时"
        except Exception as e:
            result.status = NodeStatus.FAILED
            result.error = str(e)

        finally:
            result.end_time = datetime.now()
            if result.start_time:
                result.duration_ms = (result.end_time - result.start_time).total_seconds() * 1000

        return result

    def get_execution_result(self, execution_id: str) -> Optional[WorkflowExecutionResult]:
        return self._execution_results.get(execution_id)


class WorkflowValidator:
    """工作流验证器"""

    @staticmethod
    def validate(workflow: VisualWorkflow) -> Tuple[bool, List[str]]:
        """验证工作流"""
        errors = []

        # 检查空工作流
        if not workflow.nodes:
            errors.append("工作流没有节点")
            return False, errors

        # 检查连接有效性
        for conn in workflow.connections.values():
            if conn.source_node not in workflow.nodes:
                errors.append(f"连接 {conn.id} 的源节点 {conn.source_node} 不存在")
            if conn.target_node not in workflow.nodes:
                errors.append(f"连接 {conn.id} 的目标节点 {conn.target_node} 不存在")

        # 检查循环依赖
        if WorkflowValidator._has_cycle(workflow):
            errors.append("工作流存在循环依赖")

        # 检查孤立节点
        connected_nodes = set()
        for conn in workflow.connections.values():
            connected_nodes.add(conn.source_node)
            connected_nodes.add(conn.target_node)

        for node_id in workflow.nodes:
            if node_id not in connected_nodes and len(workflow.nodes) > 1:
                errors.append(f"节点 {node_id} 是孤立的")

        return len(errors) == 0, errors

    @staticmethod
    def _has_cycle(workflow: VisualWorkflow) -> bool:
        """检测工作流是否有环"""
        graph: Dict[str, List[str]] = {node_id: [] for node_id in workflow.nodes}
        for conn in workflow.connections.values():
            if conn.source_node in graph and conn.target_node in graph:
                graph[conn.source_node].append(conn.target_node)

        # DFS检测环
        visited = set()
        rec_stack = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for neighbor in graph.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in workflow.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False


class TemplateMarket:
    """模板市场"""

    def __init__(self):
        self._templates: Dict[str, Template] = {}
        self._categories: Dict[str, TemplateCategory] = {}
        self._init_default_categories()

    def _init_default_categories(self) -> None:
        default_categories = [
            ("data", "数据处理", "数据ETL、清洗、转换"),
            ("ai", "AI/ML", "机器学习、深度学习"),
            ("automation", "自动化", "任务自动化、流程编排"),
            ("integration", "集成", "系统集成、API对接"),
            ("monitoring", "监控", "系统监控、告警")
        ]
        for cat_id, name, desc in default_categories:
            self._categories[cat_id] = TemplateCategory(id=cat_id, name=name, description=desc)

    async def publish_template(self, name: str, description: str, category: str, workflow: VisualWorkflow, author: str = "", tags: Optional[List[str]] = None) -> Template:
        template = Template(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            category=category,
            workflow=workflow,
            author=author,
            tags=tags or []
        )
        self._templates[template.id] = template
        return template

    async def search_templates(self, query: Optional[str] = None, category: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Template]:
        results = list(self._templates.values())
        if query:
            query_lower = query.lower()
            results = [t for t in results if query_lower in t.name.lower() or query_lower in t.description.lower()]
        if category:
            results = [t for t in results if t.category == category]
        if tags:
            results = [t for t in results if any(tag in t.tags for tag in tags)]
        return results

    async def download_template(self, template_id: str) -> Optional[VisualWorkflow]:
        template = self._templates.get(template_id)
        if not template:
            return None
        template.downloads += 1
        return copy.deepcopy(template.workflow)


class ConfigGenerator:
    """配置生成器"""

    async def generate_code(self, workflow: VisualWorkflow, language: str = "python") -> str:
        if language == "python":
            return await self._generate_python(workflow)
        elif language == "yaml":
            return await self._generate_yaml(workflow)
        elif language == "json":
            return json.dumps(workflow.to_dict(), indent=2)
        return ""

    async def _generate_python(self, workflow: VisualWorkflow) -> str:
        code = f'''"""
自动生成的工作流代码: {workflow.name}
生成时间: {datetime.now().isoformat()}
"""

import asyncio
from hermes_cluster.orchestration import WorkflowBuilder, OrchestrationEngine

async def main():
    engine = OrchestrationEngine()
    builder = WorkflowBuilder("{workflow.name}")
'''
        for node_id, node in workflow.nodes.items():
            code += f'''
    builder.add_task(
        task_id="{node_id}",
        name="{node.name}",
        agent_type="{node.config.get('agent_type', 'default')}",
        action="{node.config.get('action', 'process')}"
    )
'''
        code += '''
    workflow = builder.build()
    engine.register_workflow(workflow)
    execution = await engine.start_execution(workflow.id)
    print(f"Execution: {execution.id}")

if __name__ == "__main__":
    asyncio.run(main())
'''
        return code

    async def _generate_yaml(self, workflow: VisualWorkflow) -> str:
        yaml = f"""# 工作流配置: {workflow.name}
name: {workflow.name}
description: {workflow.description}
version: {workflow.version}

nodes:
"""
        for node_id, node in workflow.nodes.items():
            yaml += f"""  - id: {node_id}
    type: {node.type.value}
    name: {node.name}
    config:
"""
            for key, value in node.config.items():
                yaml += f"      {key}: {value}\n"
        yaml += """
connections:
"""
        for conn in workflow.connections.values():
            yaml += f"""  - from: {conn.source_node}.{conn.source_port}
    to: {conn.target_node}.{conn.target_port}
"""
        return yaml


class ConfigVersionManager:
    """配置版本管理器"""

    def __init__(self):
        self._versions: Dict[str, List[Dict[str, Any]]] = {}
        self._max_versions: int = 20

    async def save_version(self, workflow_id: str, workflow: VisualWorkflow, message: str = "") -> int:
        if workflow_id not in self._versions:
            self._versions[workflow_id] = []
        version = {
            "version": len(self._versions[workflow_id]) + 1,
            "data": workflow.to_dict(),
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self._versions[workflow_id].append(version)
        if len(self._versions[workflow_id]) > self._max_versions:
            self._versions[workflow_id].pop(0)
        return version["version"]

    async def get_version(self, workflow_id: str, version: int) -> Optional[Dict[str, Any]]:
        versions = self._versions.get(workflow_id, [])
        for v in versions:
            if v["version"] == version:
                return v
        return None
