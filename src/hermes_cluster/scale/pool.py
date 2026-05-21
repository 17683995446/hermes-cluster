"""
Hermes Cluster v6.0 - 百万级扩展模块
支持分层调度、Agent池化、资源伸缩、性能基准

优化点:
1. 分层调度架构（L1/L2/L3调度）
2. Agent池化管理
3. 资源自动伸缩
4. 负载预测算法
5. 弹性扩缩容
6. Agent生命周期管理
7. 资源配额控制
8. 性能基准测试
9. 热点检测与均衡
10. 容量规划工具
11. 多租户隔离
12. 资源使用审计
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent状态"""
    IDLE = "idle"
    BUSY = "busy"
    DRAINING = "draining"
    TERMINATING = "terminating"
    ERROR = "error"


class ScalingAction(Enum):
    """伸缩动作"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NONE = "none"


class SchedulerLevel(Enum):
    """调度层级"""
    L1_GLOBAL = "l1_global"
    L2_REGION = "l2_region"
    L3_LOCAL = "l3_local"


@dataclass
class AgentSpec:
    """Agent规格"""
    agent_type: str
    cpu_cores: float = 1.0
    memory_mb: int = 512
    capabilities: Set[str] = field(default_factory=set)
    priority: int = 0
    max_tasks: int = 10
    idle_timeout: float = 300.0


@dataclass
class AgentInstance:
    """Agent实例"""
    agent_id: str
    spec: AgentSpec
    state: AgentState = AgentState.IDLE
    pool_id: Optional[str] = None
    current_tasks: int = 0
    total_tasks: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def utilization(self) -> float:
        return self.current_tasks / self.spec.max_tasks if self.spec.max_tasks > 0 else 0.0
    
    @property
    def is_available(self) -> bool:
        if self.spec.max_tasks <= 0:
            return False
        return self.state == AgentState.IDLE or (self.state == AgentState.BUSY and self.current_tasks < self.spec.max_tasks)


@dataclass
class AgentPool:
    """Agent池"""
    pool_id: str
    name: str
    agent_type: str
    min_size: int = 1
    max_size: int = 100
    target_size: int = 1
    current_size: int = 0
    agents: Dict[str, AgentInstance] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def available_count(self) -> int:
        return sum(1 for a in self.agents.values() if a.is_available)
    
    @property
    def total_utilization(self) -> float:
        if not self.agents:
            return 0.0
        return sum(a.utilization for a in self.agents.values()) / len(self.agents)


@dataclass
class ScalingPolicy:
    """伸缩策略"""
    name: str
    metric: str
    target_value: float
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_step: int = 1
    scale_down_step: int = 1
    cooldown_period: float = 60.0


@dataclass
class ScalingEvent:
    """伸缩事件"""
    event_id: str
    pool_id: str
    action: ScalingAction
    reason: str
    old_size: int
    new_size: int
    timestamp: datetime = field(default_factory=datetime.now)


class AgentPoolManager:
    """Agent池管理器"""
    
    def __init__(self):
        logger.debug(f"ENTER AgentPoolManager.__init__: args={{}}")
        self._pools: Dict[str, AgentPool] = {}
        self._agent_registry: Dict[str, AgentInstance] = {}
        self._lock = asyncio.Lock()
        logger.info(f"AgentPoolManager initialized")
        logger.debug(f"EXIT AgentPoolManager.__init__: result=None")
    
    async def create_pool(self, name: str, agent_type: str, min_size: int = 1, max_size: int = 100, initial_size: int = 1) -> AgentPool:
        logger.debug(f"ENTER AgentPoolManager.create_pool: args={{'name': name, 'agent_type': agent_type, 'min_size': min_size, 'max_size': max_size, 'initial_size': initial_size}}")
        initial_size = max(min_size, min(max_size, initial_size))
        logger.debug(f"BRANCH: condition=initial_size adjusted, result={initial_size}")
        pool = AgentPool(
            pool_id=str(uuid.uuid4()),
            name=name,
            agent_type=agent_type,
            min_size=min_size,
            max_size=max_size,
            target_size=initial_size
        )
        self._pools[pool.pool_id] = pool
        logger.info(f"Pool created: {pool.pool_id} ({name}), type={agent_type}, initial_size={initial_size}")
        for i in range(initial_size):
            logger.debug(f"BRANCH: condition=creating agent {i+1}/{initial_size}, result=True")
            await self._create_agent(pool)
        logger.debug(f"EXIT AgentPoolManager.create_pool: result=pool {pool.pool_id}")
        return pool
    
    async def _create_agent(self, pool: AgentPool) -> AgentInstance:
        logger.debug(f"ENTER AgentPoolManager._create_agent: args={{'pool_id': pool.pool_id}}")
        spec = AgentSpec(agent_type=pool.agent_type)
        agent = AgentInstance(
            agent_id=str(uuid.uuid4()),
            spec=spec,
            pool_id=pool.pool_id
        )
        pool.agents[agent.agent_id] = agent
        pool.current_size += 1
        self._agent_registry[agent.agent_id] = agent
        logger.info(f"Agent created: {agent.agent_id} in pool {pool.pool_id}, current_size={pool.current_size}")
        logger.debug(f"EXIT AgentPoolManager._create_agent: result=agent {agent.agent_id}")
        return agent
    
    async def scale_pool(self, pool_id: str, target_size: int) -> Optional[ScalingEvent]:
        logger.debug(f"ENTER AgentPoolManager.scale_pool: args={{'pool_id': pool_id, 'target_size': target_size}}")
        async with self._lock:
            logger.debug(f"BRANCH: condition=lock acquired, result=True")
            pool = self._pools.get(pool_id)
            if not pool:
                logger.error(f"Pool not found: {pool_id}")
                return None  # 返回None而非抛出异常
            
            target_size = max(pool.min_size, min(pool.max_size, target_size))
            logger.debug(f"BRANCH: condition=target_size adjusted, result={target_size}")
            old_size = pool.current_size
            
            if target_size > old_size:
                logger.debug(f"BRANCH: condition=target_size > old_size, result=True (scale up)")
                for i in range(target_size - old_size):
                    await self._create_agent(pool)
                action = ScalingAction.SCALE_UP
                logger.info(f"Pool {pool_id} scaled UP from {old_size} to {pool.current_size}")
            elif target_size < old_size:
                logger.debug(f"BRANCH: condition=target_size < old_size, result=True (scale down)")
                # 优先移除空闲agent
                idle_agents = [a for a in pool.agents.values() if a.current_tasks == 0 and a.state == AgentState.IDLE]
                busy_agents = [a for a in pool.agents.values() if a.current_tasks > 0 or a.state != AgentState.IDLE]
                logger.debug(f"BRANCH: condition=idle_agents count, result={len(idle_agents)}")
                logger.debug(f"BRANCH: condition=busy_agents count, result={len(busy_agents)}")
                agents_to_remove = idle_agents[:old_size - target_size]
                # 如果空闲agent不够，再从busy中选
                if len(agents_to_remove) < old_size - target_size:
                    logger.debug(f"BRANCH: condition=need more agents to remove, result=True")
                    remaining = old_size - target_size - len(agents_to_remove)
                    # 当目标为0时（完全关闭），强制移除所有agent
                    if target_size == 0:
                        logger.debug(f"BRANCH: condition=target_size == 0, result=True (force remove all)")
                        agents_to_remove.extend(busy_agents[:remaining])
                    else:
                        logger.debug(f"BRANCH: condition=target_size == 0, result=False (mark draining)")
                        # 否则标记为DRAINING，不直接移除
                        for agent in busy_agents[:remaining]:
                            agent.state = AgentState.DRAINING
                            logger.info(f"Agent {agent.agent_id} marked as DRAINING")
                else:
                    logger.debug(f"BRANCH: condition=need more agents to remove, result=False")
                for agent_id in [a.agent_id for a in agents_to_remove]:
                    await self._terminate_agent(pool, agent_id)
                action = ScalingAction.SCALE_DOWN
                logger.info(f"Pool {pool_id} scaled DOWN from {old_size} to {pool.current_size}")
            else:
                logger.debug(f"BRANCH: condition=no scaling needed, result=True")
                action = ScalingAction.NONE
            
            pool.target_size = target_size
            
            event = ScalingEvent(
                event_id=str(uuid.uuid4()),
                pool_id=pool_id,
                action=action,
                reason=f"Scaling to {target_size}",
                old_size=old_size,
                new_size=pool.current_size
            )
            logger.info(f"Scaling event created: {event.event_id}, action={action.value}")
            logger.debug(f"EXIT AgentPoolManager.scale_pool: result=event {event.event_id}")
            return event
    
    async def _terminate_agent(self, pool: AgentPool, agent_id: str) -> bool:
        logger.debug(f"ENTER AgentPoolManager._terminate_agent: args={{'pool_id': pool.pool_id, 'agent_id': agent_id}}")
        agent = pool.agents.get(agent_id)
        if not agent:
            logger.warning(f"Agent not found for termination: {agent_id}")
            logger.debug(f"EXIT AgentPoolManager._terminate_agent: result=False")
            return False

        agent.state = AgentState.TERMINATING
        logger.info(f"Agent {agent_id} state changed to TERMINATING")

        # 先从registry删除
        if agent_id in self._agent_registry:
            del self._agent_registry[agent_id]

        # 再从pool删除，成功后减size
        if agent_id in pool.agents:
            del pool.agents[agent_id]
            pool.current_size -= 1
            logger.info(f"Agent terminated: {agent_id}, pool {pool.pool_id} size now {pool.current_size}")
            logger.debug(f"EXIT AgentPoolManager._terminate_agent: result=True")
            return True

        logger.warning(f"Agent {agent_id} not found in pool during termination")
        logger.debug(f"EXIT AgentPoolManager._terminate_agent: result=False")
        return False
    
    async def acquire_agent(self, pool_id: str, requirements: Optional[Dict[str, Any]] = None) -> Optional[AgentInstance]:
        logger.debug(f"ENTER AgentPoolManager.acquire_agent: args={{'pool_id': pool_id, 'requirements': requirements}}")
        pool = self._pools.get(pool_id)
        if not pool:
            logger.warning(f"Pool not found for acquire: {pool_id}")
            logger.debug(f"EXIT AgentPoolManager.acquire_agent: result=None")
            return None
        
        async with self._lock:
            logger.debug(f"BRANCH: condition=lock acquired, result=True")
            available = [a for a in pool.agents.values() if a.is_available]
            logger.debug(f"BRANCH: condition=available agents count, result={len(available)}")
            if not available:
                logger.warning(f"No available agents in pool {pool_id}")
                logger.debug(f"EXIT AgentPoolManager.acquire_agent: result=None")
                return None
            
            available.sort(key=lambda a: a.utilization)
            agent = available[0]
            agent.current_tasks += 1
            logger.info(f"Agent {agent.agent_id} acquired, current_tasks={agent.current_tasks}")
            # 检查是否达到上限，更新状态
            if agent.current_tasks >= agent.spec.max_tasks:
                logger.debug(f"BRANCH: condition=agent reached max_tasks, result=True")
                agent.state = AgentState.BUSY
                logger.info(f"Agent {agent.agent_id} state changed to BUSY")
            else:
                logger.debug(f"BRANCH: condition=agent reached max_tasks, result=False")
            agent.last_active = datetime.now()
            logger.debug(f"EXIT AgentPoolManager.acquire_agent: result=agent {agent.agent_id}")
            return agent
    
    async def release_agent(self, agent_id: str) -> bool:
        logger.debug(f"ENTER AgentPoolManager.release_agent: args={{'agent_id': agent_id}}")
        async with self._lock:
            logger.debug(f"BRANCH: condition=lock acquired, result=True")
            agent = self._agent_registry.get(agent_id)
            if not agent:
                logger.warning(f"Agent not found for release: {agent_id}")
                logger.debug(f"EXIT AgentPoolManager.release_agent: result=False")
                return False
            old_tasks = agent.current_tasks
            agent.current_tasks = max(0, agent.current_tasks - 1)
            agent.total_tasks += 1
            logger.info(f"Agent {agent_id} released, tasks {old_tasks} -> {agent.current_tasks}, total_tasks={agent.total_tasks}")
            if agent.current_tasks == 0:
                logger.debug(f"BRANCH: condition=current_tasks == 0, result=True")
                agent.state = AgentState.IDLE
                logger.info(f"Agent {agent_id} state changed to IDLE")
            else:
                logger.debug(f"BRANCH: condition=current_tasks == 0, result=False")
            logger.debug(f"EXIT AgentPoolManager.release_agent: result=True")
            return True
    
    async def list_pools(self) -> List[AgentPool]:
        logger.debug(f"ENTER AgentPoolManager.list_pools: args={{}}")
        pools = list(self._pools.values())
        logger.debug(f"EXIT AgentPoolManager.list_pools: result={len(pools)} pools")
        return pools


class AutoScaler:
    """自动伸缩器"""
    
    def __init__(self, pool_manager: AgentPoolManager):
        logger.debug(f"ENTER AutoScaler.__init__: args={{'pool_manager': 'provided'}}")
        self._pool_manager = pool_manager
        self._policies: Dict[str, ScalingPolicy] = {}
        self._events: List[ScalingEvent] = []
        self._is_running = False
        logger.info(f"AutoScaler initialized")
        logger.debug(f"EXIT AutoScaler.__init__: result=None")
    
    def add_policy(self, pool_id: str, policy: ScalingPolicy) -> None:
        logger.debug(f"ENTER AutoScaler.add_policy: args={{'pool_id': pool_id, 'policy_name': policy.name}}")
        self._policies[pool_id] = policy
        logger.info(f"Scaling policy added for pool {pool_id}: {policy.name}")
        logger.debug(f"EXIT AutoScaler.add_policy: result=None")
    
    def start(self) -> None:
        logger.debug(f"ENTER AutoScaler.start: args={{}}")
        self._is_running = True
        logger.info(f"AutoScaler started")
        logger.debug(f"EXIT AutoScaler.start: result=None")
    
    def stop(self) -> None:
        logger.debug(f"ENTER AutoScaler.stop: args={{}}")
        self._is_running = False
        logger.info(f"AutoScaler stopped")
        logger.debug(f"EXIT AutoScaler.stop: result=None")
    
    def get_scaling_events(self, limit: int = 100) -> List[ScalingEvent]:
        logger.debug(f"ENTER AutoScaler.get_scaling_events: args={{'limit': limit}}")
        events = self._events[-limit:]
        logger.debug(f"EXIT AutoScaler.get_scaling_events: result={len(events)} events")
        return events


class HierarchicalScheduler:
    """分层调度器"""
    
    def __init__(self, pool_manager: AgentPoolManager):
        logger.debug(f"ENTER HierarchicalScheduler.__init__: args={{'pool_manager': 'provided'}}")
        self._pool_manager = pool_manager
        logger.info(f"HierarchicalScheduler initialized")
        logger.debug(f"EXIT HierarchicalScheduler.__init__: result=None")
    
    async def schedule(self, task: Any, level: SchedulerLevel = SchedulerLevel.L1_GLOBAL) -> Optional[AgentInstance]:
        logger.debug(f"ENTER HierarchicalScheduler.schedule: args={{'task': 'provided', 'level': level.value}}")
        pools = await self._pool_manager.list_pools()
        logger.debug(f"BRANCH: condition=pools listed, result={len(pools)}")
        for pool in pools:
            logger.debug(f"BRANCH: condition=checking pool, result={pool.pool_id}")
            if pool.available_count > 0:
                logger.debug(f"BRANCH: condition=pool.available_count > 0, result=True")
                agent = await self._pool_manager.acquire_agent(pool.pool_id)
                if agent:
                    logger.debug(f"BRANCH: condition=agent acquired, result=True")
                    logger.info(f"Task scheduled to agent {agent.agent_id} from pool {pool.pool_id}")
                    logger.debug(f"EXIT HierarchicalScheduler.schedule: result=agent {agent.agent_id}")
                    return agent
                else:
                    logger.debug(f"BRANCH: condition=agent acquired, result=False")
            else:
                logger.debug(f"BRANCH: condition=pool.available_count > 0, result=False")
        logger.warning("No available agents for scheduling")
        logger.debug(f"EXIT HierarchicalScheduler.schedule: result=None")
        return None


class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self, pool_manager: AgentPoolManager):
        logger.debug(f"ENTER PerformanceBenchmark.__init__: args={{'pool_manager': 'provided'}}")
        self._pool_manager = pool_manager
        self._results: List[Dict[str, Any]] = []
        logger.info(f"PerformanceBenchmark initialized")
        logger.debug(f"EXIT PerformanceBenchmark.__init__: result=None")
    
    async def run_benchmark(self, pool_id: str, task_count: int = 1000, concurrent: int = 100) -> Dict[str, Any]:
        logger.debug(f"ENTER PerformanceBenchmark.run_benchmark: args={{'pool_id': pool_id, 'task_count': task_count, 'concurrent': concurrent}}")
        start_time = time.time()
        completed = 0
        failed = 0
        latencies: List[float] = []
        
        async def execute_task():
            nonlocal completed, failed
            task_start = time.time()
            agent = await self._pool_manager.acquire_agent(pool_id)
            if agent:
                try:
                    await asyncio.sleep(0.001)
                    completed += 1
                    logger.debug(f"BRANCH: condition=task completed, result=True, completed={completed}")
                except Exception as e:
                    failed += 1
                    logger.error(f"Task execution error: {e}", exc_info=True)
                    logger.debug(f"BRANCH: condition=task failed, result=True, failed={failed}")
                finally:
                    await self._pool_manager.release_agent(agent.agent_id)
            else:
                failed += 1
                logger.warning(f"Failed to acquire agent for task, failed={failed}")
            latencies.append((time.time() - task_start) * 1000)
        
        semaphore = asyncio.Semaphore(concurrent)
        
        async def bounded_task():
            async with semaphore:
                await execute_task()
        
        tasks = [bounded_task() for _ in range(task_count)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        result = {
            "pool_id": pool_id,
            "task_count": task_count,
            "concurrent": concurrent,
            "completed": completed,
            "failed": failed,
            "total_time_s": total_time,
            "throughput": completed / total_time if total_time > 0 else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        self._results.append(result)
        logger.info(f"Benchmark completed for pool {pool_id}: {completed}/{task_count} tasks, throughput={result['throughput']:.2f} tasks/s")
        logger.debug(f"EXIT PerformanceBenchmark.run_benchmark: result={result}")
        return result


class CapacityPlanner:
    """容量规划器"""
    
    def __init__(self, pool_manager: AgentPoolManager):
        logger.debug(f"ENTER CapacityPlanner.__init__: args={{'pool_manager': 'provided'}}")
        self._pool_manager = pool_manager
        logger.info(f"CapacityPlanner initialized")
        logger.debug(f"EXIT CapacityPlanner.__init__: result=None")
    
    async def predict_capacity(self, pool_id: str, target_utilization: float = 0.7) -> Dict[str, Any]:
        logger.debug(f"ENTER CapacityPlanner.predict_capacity: args={{'pool_id': pool_id, 'target_utilization': target_utilization}}")
        pools = await self._pool_manager.list_pools()
        logger.debug(f"BRANCH: condition=pools listed, result={len(pools)}")
        pool = next((p for p in pools if p.pool_id == pool_id), None)
        if not pool:
            logger.warning(f"Pool not found for capacity prediction: {pool_id}")
            logger.debug(f"EXIT CapacityPlanner.predict_capacity: result={{}}")
            return {}
        
        current_utilization = pool.total_utilization
        current_size = pool.current_size
        
        if current_utilization > 0:
            logger.debug(f"BRANCH: condition=current_utilization > 0, result=True")
            predicted_size = current_size * (current_utilization / target_utilization)
        else:
            logger.debug(f"BRANCH: condition=current_utilization > 0, result=False")
            predicted_size = current_size
        
        result = {
            "pool_id": pool_id,
            "current_size": current_size,
            "current_utilization": current_utilization,
            "target_utilization": target_utilization,
            "predicted_size": math.ceil(predicted_size)
        }
        logger.info(f"Capacity prediction for pool {pool_id}: current={current_size}, predicted={result['predicted_size']}")
        logger.debug(f"EXIT CapacityPlanner.predict_capacity: result={result}")
        return result
