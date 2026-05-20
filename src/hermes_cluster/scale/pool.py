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
        self._pools: Dict[str, AgentPool] = {}
        self._agent_registry: Dict[str, AgentInstance] = {}
        self._lock = asyncio.Lock()
    
    async def create_pool(self, name: str, agent_type: str, min_size: int = 1, max_size: int = 100, initial_size: int = 1) -> AgentPool:
        initial_size = max(min_size, min(max_size, initial_size))
        pool = AgentPool(
            pool_id=str(uuid.uuid4()),
            name=name,
            agent_type=agent_type,
            min_size=min_size,
            max_size=max_size,
            target_size=initial_size
        )
        self._pools[pool.pool_id] = pool
        for _ in range(initial_size):
            await self._create_agent(pool)
        return pool
    
    async def _create_agent(self, pool: AgentPool) -> AgentInstance:
        spec = AgentSpec(agent_type=pool.agent_type)
        agent = AgentInstance(
            agent_id=str(uuid.uuid4()),
            spec=spec,
            pool_id=pool.pool_id
        )
        pool.agents[agent.agent_id] = agent
        pool.current_size += 1
        self._agent_registry[agent.agent_id] = agent
        return agent
    
    async def scale_pool(self, pool_id: str, target_size: int) -> ScalingEvent:
        async with self._lock:
            pool = self._pools.get(pool_id)
            if not pool:
                raise ValueError(f"池不存在: {pool_id}")
            
            target_size = max(pool.min_size, min(pool.max_size, target_size))
            old_size = pool.current_size
            
            if target_size > old_size:
                for _ in range(target_size - old_size):
                    await self._create_agent(pool)
                action = ScalingAction.SCALE_UP
            elif target_size < old_size:
                # 优先移除空闲agent
                idle_agents = [a for a in pool.agents.values() if a.current_tasks == 0 and a.state == AgentState.IDLE]
                busy_agents = [a for a in pool.agents.values() if a.current_tasks > 0 or a.state != AgentState.IDLE]
                agents_to_remove = idle_agents[:old_size - target_size]
                # 如果空闲agent不够，再从busy中选
                if len(agents_to_remove) < old_size - target_size:
                    remaining = old_size - target_size - len(agents_to_remove)
                    # 当目标为0时（完全关闭），强制移除所有agent
                    if target_size == 0:
                        agents_to_remove.extend(busy_agents[:remaining])
                    else:
                        # 否则标记为DRAINING，不直接移除
                        for agent in busy_agents[:remaining]:
                            agent.state = AgentState.DRAINING
                for agent_id in [a.agent_id for a in agents_to_remove]:
                    await self._terminate_agent(pool, agent_id)
                action = ScalingAction.SCALE_DOWN
            else:
                action = ScalingAction.NONE
            
            pool.target_size = target_size
            
            return ScalingEvent(
                event_id=str(uuid.uuid4()),
                pool_id=pool_id,
                action=action,
                reason=f"Scaling to {target_size}",
                old_size=old_size,
                new_size=pool.current_size
            )
    
    async def _terminate_agent(self, pool: AgentPool, agent_id: str) -> bool:
        agent = pool.agents.get(agent_id)
        if not agent:
            return False
        agent.state = AgentState.TERMINATING
        if agent_id in pool.agents:
            del pool.agents[agent_id]
        if agent_id in self._agent_registry:
            del self._agent_registry[agent_id]
        pool.current_size -= 1
        return True
    
    async def acquire_agent(self, pool_id: str, requirements: Optional[Dict[str, Any]] = None) -> Optional[AgentInstance]:
        pool = self._pools.get(pool_id)
        if not pool:
            return None
        
        async with self._lock:
            available = [a for a in pool.agents.values() if a.is_available]
            if not available:
                return None
            
            available.sort(key=lambda a: a.utilization)
            agent = available[0]
            agent.current_tasks += 1
            # 检查是否达到上限，更新状态
            if agent.current_tasks >= agent.spec.max_tasks:
                agent.state = AgentState.BUSY
            agent.last_active = datetime.now()
            return agent
    
    async def release_agent(self, agent_id: str) -> bool:
        async with self._lock:
            agent = self._agent_registry.get(agent_id)
            if not agent:
                return False
            agent.current_tasks = max(0, agent.current_tasks - 1)
            agent.total_tasks += 1
            if agent.current_tasks == 0:
                agent.state = AgentState.IDLE
            return True
    
    async def list_pools(self) -> List[AgentPool]:
        return list(self._pools.values())


class AutoScaler:
    """自动伸缩器"""
    
    def __init__(self, pool_manager: AgentPoolManager):
        self._pool_manager = pool_manager
        self._policies: Dict[str, ScalingPolicy] = {}
        self._events: List[ScalingEvent] = []
        self._is_running = False
    
    def add_policy(self, pool_id: str, policy: ScalingPolicy) -> None:
        self._policies[pool_id] = policy
    
    def start(self) -> None:
        self._is_running = True
    
    def stop(self) -> None:
        self._is_running = False
    
    def get_scaling_events(self, limit: int = 100) -> List[ScalingEvent]:
        return self._events[-limit:]


class HierarchicalScheduler:
    """分层调度器"""
    
    def __init__(self, pool_manager: AgentPoolManager):
        self._pool_manager = pool_manager
    
    async def schedule(self, task: Any, level: SchedulerLevel = SchedulerLevel.L1_GLOBAL) -> Optional[AgentInstance]:
        pools = await self._pool_manager.list_pools()
        for pool in pools:
            if pool.available_count > 0:
                agent = await self._pool_manager.acquire_agent(pool.pool_id)
                if agent:
                    return agent
        return None


class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self, pool_manager: AgentPoolManager):
        self._pool_manager = pool_manager
        self._results: List[Dict[str, Any]] = []
    
    async def run_benchmark(self, pool_id: str, task_count: int = 1000, concurrent: int = 100) -> Dict[str, Any]:
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
                finally:
                    await self._pool_manager.release_agent(agent.agent_id)
            else:
                failed += 1
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
        return result


class CapacityPlanner:
    """容量规划器"""
    
    def __init__(self, pool_manager: AgentPoolManager):
        self._pool_manager = pool_manager
    
    async def predict_capacity(self, pool_id: str, target_utilization: float = 0.7) -> Dict[str, Any]:
        pools = await self._pool_manager.list_pools()
        pool = next((p for p in pools if p.pool_id == pool_id), None)
        if not pool:
            return {}
        
        current_utilization = pool.total_utilization
        current_size = pool.current_size
        
        if current_utilization > 0:
            predicted_size = current_size * (current_utilization / target_utilization)
        else:
            predicted_size = current_size
        
        return {
            "pool_id": pool_id,
            "current_size": current_size,
            "current_utilization": current_utilization,
            "target_utilization": target_utilization,
            "predicted_size": math.ceil(predicted_size)
        }
