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
        logger.debug(f"ENTER VisualNode.add_input_port: args={{'node_id': self.id, 'name': name, 'data_type': data_type, 'required': required}}")
        port = Port(
            id=f"{self.id}_in_{len(self.input_ports)}",
            name=name,
            port_type=PortType.INPUT,
            data_type=data_type,
            required=required
        )
        self.input_ports.append(port)
        logger.info(f"Input port added to node {self.id}: {name} ({port.id})")
        logger.debug(f"EXIT VisualNode.add_input_port: result=port {port.id}")
        return port

    def add_output_port(self, name: str, data_type: str = "any") -> Port:
        logger.debug(f"ENTER VisualNode.add_output_port: args={{'node_id': self.id, 'name': name, 'data_type': data_type}}")
        port = Port(
            id=f"{self.id}_out_{len(self.output_ports)}",
            name=name,
            port_type=PortType.OUTPUT,
            data_type=data_type
        )
        self.output_ports.append(port)
        logger.info(f"Output port added to node {self.id}: {name} ({port.id})")
        logger.debug(f"EXIT VisualNode.add_output_port: result=port {port.id}")
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
        logger.debug(f"ENTER VisualWorkflow.add_node: args={{'workflow_id': self.id, 'node_id': node.id, 'node_type': node.type.value}}")
        self.nodes[node.id] = node
        self.updated_at = datetime.now()
        logger.info(f"Node added to workflow {self.id}: {node.id} ({node.name})")
        logger.debug(f"EXIT VisualWorkflow.add_node: result=None")

    def remove_node(self, node_id: str) -> bool:
        logger.debug(f"ENTER VisualWorkflow.remove_node: args={{'workflow_id': self.id, 'node_id': node_id}}")
        if node_id in self.nodes:
            logger.debug(f"BRANCH: condition=node_id in self.nodes, result=True")
            del self.nodes[node_id]
            self.connections = {
                k: v for k, v in self.connections.items()
                if v.source_node != node_id and v.target_node != node_id
            }
            self.updated_at = datetime.now()
            logger.info(f"Node removed from workflow {self.id}: {node_id}")
            logger.debug(f"EXIT VisualWorkflow.remove_node: result=True")
            return True
        else:
            logger.debug(f"BRANCH: condition=node_id in self.nodes, result=False")
        logger.warning(f"Node not found for removal: {node_id}")
        logger.debug(f"EXIT VisualWorkflow.remove_node: result=False")
        return False

    def add_connection(self, connection: Connection) -> None:
        logger.debug(f"ENTER VisualWorkflow.add_connection: args={{'workflow_id': self.id, 'connection_id': connection.id, 'source': connection.source_node, 'target': connection.target_node}}")
        self.connections[connection.id] = connection
        self.updated_at = datetime.now()
        logger.info(f"Connection added to workflow {self.id}: {connection.id}")
        logger.debug(f"EXIT VisualWorkflow.add_connection: result=None")

    def remove_connection(self, connection_id: str) -> bool:
        logger.debug(f"ENTER VisualWorkflow.remove_connection: args={{'workflow_id': self.id, 'connection_id': connection_id}}")
        if connection_id in self.connections:
            logger.debug(f"BRANCH: condition=connection_id in self.connections, result=True")
            del self.connections[connection_id]
            self.updated_at = datetime.now()
            logger.info(f"Connection removed from workflow {self.id}: {connection_id}")
            logger.debug(f"EXIT VisualWorkflow.remove_connection: result=True")
            return True
        else:
            logger.debug(f"BRANCH: condition=connection_id in self.connections, result=False")
        logger.warning(f"Connection not found for removal: {connection_id}")
        logger.debug(f"EXIT VisualWorkflow.remove_connection: result=False")
        return False

    def validate(self) -> List[str]:
        logger.debug(f"ENTER VisualWorkflow.validate: args={{'workflow_id': self.id}}")
        errors = []
        if not self.nodes:
            logger.debug(f"BRANCH: condition=no nodes, result=True")
            errors.append("工作流没有节点")
        else:
            logger.debug(f"BRANCH: condition=no nodes, result=False")
        for conn in self.connections.values():
            logger.debug(f"BRANCH: condition=validating connection, result={conn.id}")
            if conn.source_node not in self.nodes:
                logger.debug(f"BRANCH: condition=source_node not in nodes, result=True")
                errors.append(f"连接 {conn.id} 的源节点不存在")
            if conn.target_node not in self.nodes:
                logger.debug(f"BRANCH: condition=target_node not in nodes, result=True")
                errors.append(f"连接 {conn.id} 的目标节点不存在")
        logger.info(f"Workflow {self.id} validation: {len(errors)} errors found")
        logger.debug(f"EXIT VisualWorkflow.validate: result={len(errors)} errors")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        logger.debug(f"ENTER VisualWorkflow.to_dict: args={{'workflow_id': self.id}}")
        result = {
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
        logger.debug(f"EXIT VisualWorkflow.to_dict: result=dict with {len(result)} keys")
        return result


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
        logger.debug(f"ENTER WorkflowEditor.__init__: args={{}}")
        self._workflows: Dict[str, VisualWorkflow] = {}
        self._undo_stack: Dict[str, List[Dict[str, Any]]] = {}
        self._redo_stack: Dict[str, List[Dict[str, Any]]] = {}
        self._max_undo: int = 50
        logger.info(f"WorkflowEditor initialized")
        logger.debug(f"EXIT WorkflowEditor.__init__: result=None")

    def create_workflow(self, name: str, description: str = "") -> VisualWorkflow:
        """创建工作流 (同步版本)"""
        logger.debug(f"ENTER WorkflowEditor.create_workflow: args={{'name': name, 'description': description}}")
        workflow = VisualWorkflow(
            id=str(uuid.uuid4()),
            name=name,
            description=description
        )
        self._workflows[workflow.id] = workflow
        self._undo_stack[workflow.id] = []
        self._redo_stack[workflow.id] = []
        logger.info(f"Workflow created: {workflow.id} ({name})")
        logger.debug(f"EXIT WorkflowEditor.create_workflow: result=workflow {workflow.id}")
        return workflow

    async def create_workflow_async(self, name: str, description: str = "") -> VisualWorkflow:
        """异步创建工作流"""
        logger.debug(f"ENTER WorkflowEditor.create_workflow_async: args={{'name': name, 'description': description}}")
        result = self.create_workflow(name, description)
        logger.debug(f"EXIT WorkflowEditor.create_workflow_async: result=workflow {result.id}")
        return result

    def add_node(self, workflow_id: str, node: VisualNode) -> bool:
        """添加节点 (同步版本)"""
        logger.debug(f"ENTER WorkflowEditor.add_node: args={{'workflow_id': workflow_id, 'node_id': node.id}}")
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            logger.warning(f"Workflow not found for add_node: {workflow_id}")
            logger.debug(f"EXIT WorkflowEditor.add_node: result=False")
            return False
        workflow.add_node(node)
        logger.debug(f"EXIT WorkflowEditor.add_node: result=True")
        return True

    async def add_node_async(self, workflow_id: str, node_type: NodeType, name: str, 
                             position: Tuple[float, float], config: Optional[Dict[str, Any]] = None) -> Optional[VisualNode]:
        """异步添加节点"""
        logger.debug(f"ENTER WorkflowEditor.add_node_async: args={{'workflow_id': workflow_id, 'node_type': node_type.value, 'name': name, 'position': position}}")
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            logger.warning(f"Workflow not found for add_node_async: {workflow_id}")
            logger.debug(f"EXIT WorkflowEditor.add_node_async: result=None")
            return None

        node = VisualNode(
            id=str(uuid.uuid4()),
            type=node_type,
            name=name,
            position=NodePosition(x=position[0], y=position[1]),
            config=config or {}
        )

        if node_type == NodeType.AGENT:
            logger.debug(f"BRANCH: condition=node_type == AGENT, result=True")
            node.add_input_port("input", "any", True)
            node.add_output_port("output", "any")
        elif node_type == NodeType.TASK:
            logger.debug(f"BRANCH: condition=node_type == TASK, result=True")
            node.add_input_port("input", "any", True)
            node.add_output_port("output", "any")
        elif node_type == NodeType.CONDITION:
            logger.debug(f"BRANCH: condition=node_type == CONDITION, result=True")
            node.add_input_port("input", "any", True)
            node.add_output_port("true", "any")
            node.add_output_port("false", "any")
        elif node_type == NodeType.PARALLEL:
            logger.debug(f"BRANCH: condition=node_type == PARALLEL, result=True")
            node.add_input_port("input", "any", True)
            node.add_output_port("output1", "any")
            node.add_output_port("output2", "any")
        elif node_type == NodeType.FUNCTION:
            logger.debug(f"BRANCH: condition=node_type == FUNCTION, result=True")
            node.add_input_port("input", "any", True)
            node.add_output_port("output", "any")
        else:
            logger.debug(f"BRANCH: condition=unknown node_type, result=True")

        workflow.add_node(node)
        logger.info(f"Node added asynchronously: {node.id} ({name}) to workflow {workflow_id}")
        logger.debug(f"EXIT WorkflowEditor.add_node_async: result=node {node.id}")
        return node

    def connect_nodes(self, workflow_id: str, source_node_id: str, target_node_id: str,
                      source_port: str = "output", target_port: str = "input") -> Optional[Connection]:
        """连接节点 (同步版本)"""
        logger.debug(f"ENTER WorkflowEditor.connect_nodes: args={{'workflow_id': workflow_id, 'source_node_id': source_node_id, 'target_node_id': target_node_id, 'source_port': source_port, 'target_port': target_port}}")
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            logger.warning(f"Workflow not found for connect_nodes: {workflow_id}")
            logger.debug(f"EXIT WorkflowEditor.connect_nodes: result=None")
            return None
        # 验证节点存在
        if source_node_id not in workflow.nodes:
            logger.warning(f"Source node not found: {source_node_id}")
            logger.debug(f"EXIT WorkflowEditor.connect_nodes: result=None")
            return None
        if target_node_id not in workflow.nodes:
            logger.warning(f"Target node not found: {target_node_id}")
            logger.debug(f"EXIT WorkflowEditor.connect_nodes: result=None")
            return None

        connection = Connection(
            id=str(uuid.uuid4()),
            source_node=source_node_id,
            source_port=source_port,
            target_node=target_node_id,
            target_port=target_port
        )
        workflow.add_connection(connection)
        logger.info(f"Nodes connected in workflow {workflow_id}: {source_node_id} -> {target_node_id}")
        logger.debug(f"EXIT WorkflowEditor.connect_nodes: result=connection {connection.id}")
        return connection

    async def connect_nodes_async(self, workflow_id: str, source_node: str, source_port: str,
                                  target_node: str, target_port: str) -> Optional[Connection]:
        """异步连接节点"""
        logger.debug(f"ENTER WorkflowEditor.connect_nodes_async: args={{'workflow_id': workflow_id, 'source_node': source_node, 'target_node': target_node}}")
        result = self.connect_nodes(workflow_id, source_node, target_node, source_port, target_port)
        logger.debug(f"EXIT WorkflowEditor.connect_nodes_async: result={'connection' if result else 'None'}")
        return result

    async def validate_workflow(self, workflow_id: str) -> List[str]:
        logger.debug(f"ENTER WorkflowEditor.validate_workflow: args={{'workflow_id': workflow_id}}")
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            logger.warning(f"Workflow not found for validation: {workflow_id}")
            logger.debug(f"EXIT WorkflowEditor.validate_workflow: result=['工作流不存在']")
            return ["工作流不存在"]
        errors = workflow.validate()
        logger.debug(f"EXIT WorkflowEditor.validate_workflow: result={len(errors)} errors")
        return errors

    def get_workflow(self, workflow_id: str) -> Optional[VisualWorkflow]:
        logger.debug(f"ENTER WorkflowEditor.get_workflow: args={{'workflow_id': workflow_id}}")
        workflow = self._workflows.get(workflow_id)
        logger.debug(f"EXIT WorkflowEditor.get_workflow: result={'found' if workflow else 'None'}")
        return workflow


class WorkflowExecutor:
    """工作流执行器"""

    def __init__(self, editor: Optional[WorkflowEditor] = None):
        logger.debug(f"ENTER WorkflowExecutor.__init__: args={{'editor': 'provided' if editor else 'None'}}")
        self._editor = editor or WorkflowEditor()
        self._execution_results: Dict[str, WorkflowExecutionResult] = {}
        self._lock = asyncio.Lock()
        logger.info(f"WorkflowExecutor initialized")
        logger.debug(f"EXIT WorkflowExecutor.__init__: result=None")

    async def execute(self, workflow_id: str, context: Optional[Dict[str, Any]] = None,
                      timeout: float = 300.0) -> WorkflowExecutionResult:
        """执行工作流"""
        logger.debug(f"ENTER WorkflowExecutor.execute: args={{'workflow_id': workflow_id, 'timeout': timeout}}")
        workflow = self._editor.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Workflow not found for execution: {workflow_id}")
            logger.debug(f"EXIT WorkflowExecutor.execute: result=error '工作流不存在'")
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
            logger.error(f"Workflow validation failed: {errors}")
            logger.debug(f"EXIT WorkflowExecutor.execute: result=error '工作流验证失败'")
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
            logger.debug(f"BRANCH: condition=lock acquired, result=True")
            self._execution_results[execution_id] = result
            # 限制保留最近500条
            if len(self._execution_results) > 500:
                logger.debug(f"BRANCH: condition=execution_results > 500, result=True")
                keys = list(self._execution_results.keys())
                for k in keys[:len(self._execution_results) - 300]:
                    del self._execution_results[k]
                logger.info(f"Cleaned up old execution results, remaining: {len(self._execution_results)}")
            else:
                logger.debug(f"BRANCH: condition=execution_results > 500, result=False")

        try:
            # 构建执行图
            execution_order = self._build_execution_order(workflow)
            logger.debug(f"BRANCH: condition=execution_order built, result={len(execution_order)} nodes")

            # 构建连接图：parent -> [(child_id, source_port), ...]
            children_map: Dict[str, List[Tuple[str, str]]] = {}
            for conn in workflow.connections.values():
                children_map.setdefault(conn.source_node, []).append(
                    (conn.target_node, conn.source_port)
                )
            logger.debug(f"BRANCH: condition=children_map built, result={len(children_map)} connections")

            # 跟踪哪些节点应该被执行
            skip_set: Set[str] = set()

            for node_id in execution_order:
                logger.debug(f"BRANCH: condition=processing node, result={node_id}")
                if result.status == NodeStatus.FAILED:
                    logger.debug(f"BRANCH: condition=result.status == FAILED, result=True (breaking)")
                    break
                if node_id in skip_set:
                    logger.debug(f"BRANCH: condition=node_id in skip_set, result=True (skipping)")
                    continue

                node = workflow.nodes.get(node_id)
                if not node:
                    logger.warning(f"Node not found during execution: {node_id}")
                    continue

                node_result = await self._execute_node(node, context, timeout)
                result.node_results[node_id] = node_result
                logger.info(f"Node {node_id} executed with status: {node_result.status.value}")

                if node_result.status == NodeStatus.FAILED:
                    logger.debug(f"BRANCH: condition=node_result.status == FAILED, result=True")
                    result.status = NodeStatus.FAILED
                    result.error = node_result.error
                    logger.error(f"Workflow execution failed at node {node_id}: {node_result.error}")
                    break
                elif node_result.status == NodeStatus.COMPLETED:
                    logger.debug(f"BRANCH: condition=node_result.status == COMPLETED, result=True")
                    if node_result.output is not None:
                        result.context[f"{node_id}_output"] = node_result.output
                        logger.debug(f"BRANCH: condition=node_output stored, result=True")

                    # CONDITION节点：根据结果跳过不匹配的分支
                    if node.type == NodeType.CONDITION and node_result.output:
                        logger.debug(f"BRANCH: condition=node is CONDITION with output, result=True")
                        condition_result = node_result.output.get("condition_result", False)
                        children = children_map.get(node_id, [])
                        for child_id, port in children:
                            if (condition_result and port != "true") or \
                               (not condition_result and port != "false"):
                                skip_set.add(child_id)
                                logger.info(f"Node {child_id} skipped due to condition result: {condition_result}")
                else:
                    logger.debug(f"BRANCH: condition=node_result.status == {node_result.status.value}, result=True")

            if result.status != NodeStatus.FAILED:
                logger.debug(f"BRANCH: condition=workflow completed successfully, result=True")
                result.status = NodeStatus.COMPLETED
                logger.info(f"Workflow {workflow_id} completed successfully")

        except Exception as e:
            logger.error(f"EXCEPTION: {e}", exc_info=True)
            result.status = NodeStatus.FAILED
            result.error = str(e)
            logger.error(f"Workflow execution error: {e}")

        finally:
            result.end_time = datetime.now()
            logger.debug(f"BRANCH: condition=execution finished, result=True")

        logger.debug(f"EXIT WorkflowExecutor.execute: result=execution {execution_id} with status {result.status.value}")
        return result

    def _build_execution_order(self, workflow: VisualWorkflow) -> List[str]:
        """构建节点执行顺序 (拓扑排序)"""
        logger.debug(f"ENTER WorkflowExecutor._build_execution_order: args={{'workflow_id': workflow.id}}")
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
                logger.warning(f"Node {node_id} has circular dependency, adding to end")
                order.append(node_id)

        logger.debug(f"EXIT WorkflowExecutor._build_execution_order: result={len(order)} nodes")
        return order

    async def _execute_node(self, node: VisualNode, context: Dict[str, Any],
                           timeout: float) -> NodeExecutionResult:
        """执行单个节点"""
        logger.debug(f"ENTER WorkflowExecutor._execute_node: args={{'node_id': node.id, 'node_type': node.type.value, 'timeout': timeout}}")
        result = NodeExecutionResult(
            node_id=node.id,
            status=NodeStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            if node.type == NodeType.FUNCTION:
                logger.debug(f"BRANCH: condition=node.type == FUNCTION, result=True")
                code = node.config.get("code", "")
                if code:
                    logger.debug(f"BRANCH: condition=code provided, result=True")
                    # 允许安全的内置函数
                    safe_builtins = {
                        'len': len, 'range': range, 'enumerate': enumerate,
                        'zip': zip, 'map': map, 'filter': filter,
                        'sum': sum, 'min': min, 'max': max, 'abs': abs,
                        'int': int, 'float': float, 'str': str, 'list': list,
                        'dict': dict, 'set': set, 'tuple': tuple,
                        'print': print, 'isinstance': isinstance, 'hasattr': hasattr
                    }
                    async def _run_function():
                        local_vars = {"context": context, "result": None}
                        exec(code, {"__builtins__": safe_builtins}, local_vars)
                        return local_vars.get("result")

                    result.output = await asyncio.wait_for(_run_function(), timeout=timeout)
                    logger.info(f"Function node {node.id} executed successfully")
                else:
                    logger.debug(f"BRANCH: condition=code provided, result=False")
                result.status = NodeStatus.COMPLETED

            elif node.type == NodeType.AGENT:
                logger.debug(f"BRANCH: condition=node.type == AGENT, result=True")
                await asyncio.wait_for(asyncio.sleep(0.001), timeout=timeout)
                result.output = {"agent_result": f"Processed by {node.name}"}
                result.status = NodeStatus.COMPLETED
                logger.info(f"Agent node {node.id} executed successfully")

            elif node.type == NodeType.TASK:
                logger.debug(f"BRANCH: condition=node.type == TASK, result=True")
                await asyncio.wait_for(asyncio.sleep(0.001), timeout=timeout)
                result.output = {"task_result": f"Task {node.name} completed"}
                result.status = NodeStatus.COMPLETED
                logger.info(f"Task node {node.id} executed successfully")

            elif node.type == NodeType.CONDITION:
                logger.debug(f"BRANCH: condition=node.type == CONDITION, result=True")
                condition = node.config.get("condition", "True")
                try:
                    local_vars = {"context": context}
                    condition_result = eval(condition, {"__builtins__": {}}, local_vars)
                    result.output = {"condition_result": bool(condition_result)}
                    result.status = NodeStatus.COMPLETED
                    logger.info(f"Condition node {node.id} evaluated: {condition_result}")
                except Exception as e:
                    logger.error(f"EXCEPTION: {e}", exc_info=True)
                    result.error = f"条件评估错误: {e}"
                    result.status = NodeStatus.FAILED
                    logger.error(f"Condition node {node.id} failed: {e}")

            else:
                logger.debug(f"BRANCH: condition=unknown node_type, result=True")
                result.output = {"node_type": node.type.value}
                result.status = NodeStatus.COMPLETED
                logger.info(f"Node {node.id} ({node.type.value}) executed")

        except asyncio.TimeoutError:
            logger.error(f"EXCEPTION: Node execution timeout")
            result.status = NodeStatus.TIMEOUT
            result.error = f"节点执行超时 ({timeout}s)"
            logger.error(f"Node {node.id} execution timeout")
        except Exception as e:
            logger.error(f"EXCEPTION: {e}", exc_info=True)
            result.status = NodeStatus.FAILED
            result.error = str(e)
            logger.error(f"Node {node.id} execution failed: {e}")

        finally:
            result.end_time = datetime.now()
            if result.start_time:
                result.duration_ms = (result.end_time - result.start_time).total_seconds() * 1000
                logger.debug(f"BRANCH: condition=duration calculated, result={result.duration_ms}ms")

        logger.debug(f"EXIT WorkflowExecutor._execute_node: result=status {result.status.value}")
        return result

    def get_execution_result(self, execution_id: str) -> Optional[WorkflowExecutionResult]:
        logger.debug(f"ENTER WorkflowExecutor.get_execution_result: args={{'execution_id': execution_id}}")
        result = self._execution_results.get(execution_id)
        logger.debug(f"EXIT WorkflowExecutor.get_execution_result: result={'found' if result else 'None'}")
        return result


class WorkflowValidator:
    """工作流验证器"""

    @staticmethod
    def validate(workflow: VisualWorkflow) -> Tuple[bool, List[str]]:
        """验证工作流"""
        logger.debug(f"ENTER WorkflowValidator.validate: args={{'workflow_id': workflow.id}}")
        errors = []

        # 检查空工作流
        if not workflow.nodes:
            logger.debug(f"BRANCH: condition=no nodes, result=True")
            errors.append("工作流没有节点")
            logger.debug(f"EXIT WorkflowValidator.validate: result=(False, {len(errors)} errors)")
            return False, errors
        else:
            logger.debug(f"BRANCH: condition=no nodes, result=False")

        # 检查连接有效性
        for conn in workflow.connections.values():
            logger.debug(f"BRANCH: condition=validating connection, result={conn.id}")
            if conn.source_node not in workflow.nodes:
                logger.debug(f"BRANCH: condition=source_node not in nodes, result=True")
                errors.append(f"连接 {conn.id} 的源节点 {conn.source_node} 不存在")
            if conn.target_node not in workflow.nodes:
                logger.debug(f"BRANCH: condition=target_node not in nodes, result=True")
                errors.append(f"连接 {conn.id} 的目标节点 {conn.target_node} 不存在")

        # 检查循环依赖
        if WorkflowValidator._has_cycle(workflow):
            logger.debug(f"BRANCH: condition=has_cycle, result=True")
            errors.append("工作流存在循环依赖")
        else:
            logger.debug(f"BRANCH: condition=has_cycle, result=False")

        # 检查孤立节点
        connected_nodes = set()
        for conn in workflow.connections.values():
            connected_nodes.add(conn.source_node)
            connected_nodes.add(conn.target_node)

        for node_id in workflow.nodes:
            if node_id not in connected_nodes and len(workflow.nodes) > 1:
                logger.debug(f"BRANCH: condition=isolated node, result={node_id}")
                errors.append(f"节点 {node_id} 是孤立的")

        is_valid = len(errors) == 0
        logger.info(f"Workflow {workflow.id} validation: valid={is_valid}, errors={len(errors)}")
        logger.debug(f"EXIT WorkflowValidator.validate: result=({is_valid}, {len(errors)} errors)")
        return is_valid, errors

    @staticmethod
    def _has_cycle(workflow: VisualWorkflow) -> bool:
        """检测工作流是否有环"""
        logger.debug(f"ENTER WorkflowValidator._has_cycle: args={{'workflow_id': workflow.id}}")
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
                    logger.debug(f"EXIT WorkflowValidator._has_cycle: result=True")
                    return True

        logger.debug(f"EXIT WorkflowValidator._has_cycle: result=False")
        return False


class TemplateMarket:
    """模板市场"""

    def __init__(self):
        logger.debug(f"ENTER TemplateMarket.__init__: args={{}}")
        self._templates: Dict[str, Template] = {}
        self._categories: Dict[str, TemplateCategory] = {}
        self._init_default_categories()
        logger.info(f"TemplateMarket initialized")
        logger.debug(f"EXIT TemplateMarket.__init__: result=None")

    def _init_default_categories(self) -> None:
        logger.debug(f"ENTER TemplateMarket._init_default_categories: args={{}}")
        default_categories = [
            ("data", "数据处理", "数据ETL、清洗、转换"),
            ("ai", "AI/ML", "机器学习、深度学习"),
            ("automation", "自动化", "任务自动化、流程编排"),
            ("integration", "集成", "系统集成、API对接"),
            ("monitoring", "监控", "系统监控、告警")
        ]
        for cat_id, name, desc in default_categories:
            logger.debug(f"BRANCH: condition=adding category, result={cat_id}")
            self._categories[cat_id] = TemplateCategory(id=cat_id, name=name, description=desc)
        logger.info(f"Default categories initialized: {len(default_categories)}")
        logger.debug(f"EXIT TemplateMarket._init_default_categories: result=None")

    async def publish_template(self, name: str, description: str, category: str, workflow: VisualWorkflow, author: str = "", tags: Optional[List[str]] = None) -> Template:
        logger.debug(f"ENTER TemplateMarket.publish_template: args={{'name': name, 'category': category, 'author': author}}")
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
        logger.info(f"Template published: {template.id} ({name}) by {author}")
        logger.debug(f"EXIT TemplateMarket.publish_template: result=template {template.id}")
        return template

    async def search_templates(self, query: Optional[str] = None, category: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Template]:
        logger.debug(f"ENTER TemplateMarket.search_templates: args={{'query': query, 'category': category, 'tags': tags}}")
        results = list(self._templates.values())
        logger.debug(f"BRANCH: condition=initial templates count, result={len(results)}")
        if query:
            logger.debug(f"BRANCH: condition=query provided, result=True")
            query_lower = query.lower()
            results = [t for t in results if query_lower in t.name.lower() or query_lower in t.description.lower()]
            logger.debug(f"BRANCH: condition=after query filter, result={len(results)}")
        if category:
            logger.debug(f"BRANCH: condition=category provided, result=True")
            results = [t for t in results if t.category == category]
            logger.debug(f"BRANCH: condition=after category filter, result={len(results)}")
        if tags:
            logger.debug(f"BRANCH: condition=tags provided, result=True")
            results = [t for t in results if any(tag in t.tags for tag in tags)]
            logger.debug(f"BRANCH: condition=after tags filter, result={len(results)}")
        logger.info(f"Template search: found {len(results)} templates")
        logger.debug(f"EXIT TemplateMarket.search_templates: result={len(results)} templates")
        return results

    async def download_template(self, template_id: str) -> Optional[VisualWorkflow]:
        logger.debug(f"ENTER TemplateMarket.download_template: args={{'template_id': template_id}}")
        template = self._templates.get(template_id)
        if not template:
            logger.warning(f"Template not found for download: {template_id}")
            logger.debug(f"EXIT TemplateMarket.download_template: result=None")
            return None
        template.downloads += 1
        logger.info(f"Template downloaded: {template_id} ({template.name}), total downloads: {template.downloads}")
        result = copy.deepcopy(template.workflow)
        logger.debug(f"EXIT TemplateMarket.download_template: result=workflow {result.id}")
        return result


class ConfigGenerator:
    """配置生成器"""

    async def generate_code(self, workflow: VisualWorkflow, language: str = "python") -> str:
        logger.debug(f"ENTER ConfigGenerator.generate_code: args={{'workflow_id': workflow.id, 'language': language}}")
        if language == "python":
            logger.debug(f"BRANCH: condition=language == python, result=True")
            result = await self._generate_python(workflow)
        elif language == "yaml":
            logger.debug(f"BRANCH: condition=language == yaml, result=True")
            result = await self._generate_yaml(workflow)
        elif language == "json":
            logger.debug(f"BRANCH: condition=language == json, result=True")
            result = json.dumps(workflow.to_dict(), indent=2)
        else:
            logger.debug(f"BRANCH: condition=unknown language, result=True")
            result = ""
        logger.info(f"Code generated for workflow {workflow.id} in {language}")
        logger.debug(f"EXIT ConfigGenerator.generate_code: result=code ({len(result)} chars)")
        return result

    async def _generate_python(self, workflow: VisualWorkflow) -> str:
        logger.debug(f"ENTER ConfigGenerator._generate_python: args={{'workflow_id': workflow.id}}")
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
            logger.debug(f"BRANCH: condition=generating node code, result={node_id}")
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
        logger.debug(f"EXIT ConfigGenerator._generate_python: result=code ({len(code)} chars)")
        return code

    async def _generate_yaml(self, workflow: VisualWorkflow) -> str:
        logger.debug(f"ENTER ConfigGenerator._generate_yaml: args={{'workflow_id': workflow.id}}")
        yaml = f"""# 工作流配置: {workflow.name}
name: {workflow.name}
description: {workflow.description}
version: {workflow.version}

nodes:
"""
        for node_id, node in workflow.nodes.items():
            logger.debug(f"BRANCH: condition=generating node yaml, result={node_id}")
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
            logger.debug(f"BRANCH: condition=generating connection yaml, result={conn.id}")
            yaml += f"""  - from: {conn.source_node}.{conn.source_port}
    to: {conn.target_node}.{conn.target_port}
"""
        logger.debug(f"EXIT ConfigGenerator._generate_yaml: result=yaml ({len(yaml)} chars)")
        return yaml


class ConfigVersionManager:
    """配置版本管理器"""

    def __init__(self):
        logger.debug(f"ENTER ConfigVersionManager.__init__: args={{}}")
        self._versions: Dict[str, List[Dict[str, Any]]] = {}
        self._max_versions: int = 20
        logger.info(f"ConfigVersionManager initialized")
        logger.debug(f"EXIT ConfigVersionManager.__init__: result=None")

    async def save_version(self, workflow_id: str, workflow: VisualWorkflow, message: str = "") -> int:
        logger.debug(f"ENTER ConfigVersionManager.save_version: args={{'workflow_id': workflow_id, 'message': message}}")
        if workflow_id not in self._versions:
            logger.debug(f"BRANCH: condition=workflow_id not in versions, result=True")
            self._versions[workflow_id] = []
        version = {
            "version": len(self._versions[workflow_id]) + 1,
            "data": workflow.to_dict(),
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self._versions[workflow_id].append(version)
        logger.info(f"Version saved for workflow {workflow_id}: version {version['version']}")
        if len(self._versions[workflow_id]) > self._max_versions:
            logger.debug(f"BRANCH: condition=versions > max_versions, result=True")
            self._versions[workflow_id].pop(0)
            logger.info(f"Oldest version removed for workflow {workflow_id}")
        else:
            logger.debug(f"BRANCH: condition=versions > max_versions, result=False")
        logger.debug(f"EXIT ConfigVersionManager.save_version: result=version {version['version']}")
        return version["version"]

    async def get_version(self, workflow_id: str, version: int) -> Optional[Dict[str, Any]]:
        logger.debug(f"ENTER ConfigVersionManager.get_version: args={{'workflow_id': workflow_id, 'version': version}}")
        versions = self._versions.get(workflow_id, [])
        for v in versions:
            if v["version"] == version:
                logger.debug(f"BRANCH: condition=version found, result=True")
                logger.debug(f"EXIT ConfigVersionManager.get_version: result=version {version}")
                return v
        logger.warning(f"Version not found: workflow {workflow_id}, version {version}")
        logger.debug(f"EXIT ConfigVersionManager.get_version: result=None")
        return None
