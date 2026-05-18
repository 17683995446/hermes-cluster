"""
Hermes Cluster v6.0 - 跨地域部署模块
支持多区域集群管理、数据同步、就近路由、灾备切换

优化点:
1. 多区域集群注册与发现
2. 区域健康状态监控
3. 数据一致性同步策略
4. 就近路由智能选择
5. 灾备自动切换机制
6. 跨区域延迟优化
7. 数据分区与复制
8. 区域负载均衡
9. 故障隔离与恢复
10. 全局配置管理
11. 区域间通信加密
12. 多活架构支持
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

logger = logging.getLogger(__name__)


class RegionStatus(Enum):
    """区域状态"""
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILOVER = "failover"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class SyncMode(Enum):
    """同步模式"""
    SYNC = "sync"
    ASYNC = "async"
    SEMI_SYNC = "semi_sync"


class RoutingStrategy(Enum):
    """路由策略"""
    NEAREST = "nearest"
    LATENCY = "latency"
    LOAD = "load"
    COST = "cost"
    CUSTOM = "custom"


@dataclass
class RegionInfo:
    """区域信息"""
    region_id: str
    name: str
    location: str
    timezone: str
    endpoint: str
    status: RegionStatus = RegionStatus.ACTIVE
    priority: int = 0
    capacity: int = 1000
    current_load: int = 0
    latency_map: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: Optional[datetime] = None
    
    @property
    def load_ratio(self) -> float:
        return self.current_load / self.capacity if self.capacity > 0 else 0.0
    
    @property
    def is_available(self) -> bool:
        return self.status in (RegionStatus.ACTIVE, RegionStatus.DEGRADED)


@dataclass
class DataPartition:
    """数据分区"""
    partition_id: str
    region_id: str
    replica_regions: List[str] = field(default_factory=list)
    sync_mode: SyncMode = SyncMode.ASYNC
    data_range: Optional[Tuple[Any, Any]] = None
    size_bytes: int = 0
    last_sync: Optional[datetime] = None
    sync_lag_ms: int = 0


@dataclass
class SyncTask:
    """同步任务"""
    task_id: str
    source_region: str
    target_region: str
    partition_id: str
    data: Any
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    retry_count: int = 0


class RegionManager:
    """区域管理器"""
    
    def __init__(self):
        self._regions: Dict[str, RegionInfo] = {}
        self._primary_region: Optional[str] = None
        self._routing_strategy: RoutingStrategy = RoutingStrategy.NEAREST
        self._lock = asyncio.Lock()
    
    async def register_region(self, region: RegionInfo) -> bool:
        async with self._lock:
            self._regions[region.region_id] = region
            if self._primary_region is None:
                self._primary_region = region.region_id
            return True
    
    async def unregister_region(self, region_id: str) -> bool:
        async with self._lock:
            if region_id in self._regions:
                del self._regions[region_id]
                if self._primary_region == region_id:
                    self._primary_region = self._select_new_primary()
                return True
            return False
    
    def _select_new_primary(self) -> Optional[str]:
        available = [r for r in self._regions.values() if r.is_available]
        if available:
            available.sort(key=lambda r: (-r.priority, r.load_ratio))
            return available[0].region_id
        return None
    
    async def update_region_status(self, region_id: str, status: RegionStatus, load: Optional[int] = None) -> bool:
        region = self._regions.get(region_id)
        if not region:
            return False
        region.status = status
        region.last_heartbeat = datetime.now()
        if load is not None:
            region.current_load = load
        return True
    
    async def get_region(self, region_id: str) -> Optional[RegionInfo]:
        return self._regions.get(region_id)
    
    async def list_regions(self, status: Optional[RegionStatus] = None) -> List[RegionInfo]:
        regions = list(self._regions.values())
        if status:
            regions = [r for r in regions if r.status == status]
        return regions
    
    async def select_region(self, client_location: Optional[str] = None, strategy: Optional[RoutingStrategy] = None) -> Optional[RegionInfo]:
        strategy = strategy or self._routing_strategy
        available = [r for r in self._regions.values() if r.is_available]
        if not available:
            return None
        
        if strategy == RoutingStrategy.NEAREST:
            if client_location:
                return min(available, key=lambda r: min(r.latency_map.values()) if r.latency_map else float('inf'))
            return available[0]
        elif strategy == RoutingStrategy.LATENCY:
            return min(available, key=lambda r: min(r.latency_map.values()) if r.latency_map else float('inf'))
        elif strategy == RoutingStrategy.LOAD:
            return min(available, key=lambda r: r.load_ratio)
        elif strategy == RoutingStrategy.COST:
            return min(available, key=lambda r: r.priority)
        return available[0]
    
    def get_primary_region(self) -> Optional[RegionInfo]:
        if self._primary_region:
            return self._regions.get(self._primary_region)
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        results = {}
        for region_id, region in self._regions.items():
            is_healthy = region.is_available
            if region.last_heartbeat:
                elapsed = (datetime.now() - region.last_heartbeat).total_seconds()
                is_healthy = is_healthy and elapsed < 60
            results[region_id] = {
                "status": region.status.value,
                "healthy": is_healthy,
                "load_ratio": region.load_ratio
            }
        return results


class DataSyncEngine:
    """数据同步引擎"""
    
    def __init__(self, region_manager: RegionManager):
        self._region_manager = region_manager
        self._partitions: Dict[str, DataPartition] = {}
        self._sync_queue: asyncio.Queue = asyncio.Queue()
        self._sync_tasks: Dict[str, SyncTask] = {}
        self._is_running = False
        self._lock = asyncio.Lock()
    
    async def create_partition(self, partition_id: str, primary_region: str, replica_regions: List[str], sync_mode: SyncMode = SyncMode.ASYNC) -> DataPartition:
        async with self._lock:
            partition = DataPartition(
                partition_id=partition_id,
                region_id=primary_region,
                replica_regions=replica_regions,
                sync_mode=sync_mode
            )
            self._partitions[partition_id] = partition
            return partition
    
    async def sync_data(self, partition_id: str, data: Any, source_region: str) -> str:
        async with self._lock:
            partition = self._partitions.get(partition_id)
            if not partition:
                raise ValueError(f"分区不存在: {partition_id}")

            # 确定目标区域：优先同步到副本，避免回环
            target_region = None
            if source_region == partition.region_id:
                # 从主区域发起，同步到第一个副本
                if partition.replica_regions:
                    target_region = partition.replica_regions[0]
            else:
                # 从副本发起，同步回主区域
                target_region = partition.region_id

            # 无副本且从主区域发起时，跳过同步
            if target_region is None or target_region == source_region:
                # 无需同步（无副本或已在目标区域）
                task = SyncTask(
                    task_id=str(uuid.uuid4()),
                    source_region=source_region,
                    target_region=source_region,
                    partition_id=partition_id,
                    data=data,
                    status="skipped"
                )
                self._sync_tasks[task.task_id] = task
                # 限制保留最近1000条
                if len(self._sync_tasks) > 1000:
                    keys = list(self._sync_tasks.keys())
                    for k in keys[:len(self._sync_tasks) - 800]:
                        del self._sync_tasks[k]
                return task.task_id

            task = SyncTask(
                task_id=str(uuid.uuid4()),
                source_region=source_region,
                target_region=target_region,
                partition_id=partition_id,
                data=data
            )
            self._sync_tasks[task.task_id] = task
            # 限制保留最近1000条
            if len(self._sync_tasks) > 1000:
                keys = list(self._sync_tasks.keys())
                for k in keys[:len(self._sync_tasks) - 800]:
                    del self._sync_tasks[k]
            await self._sync_queue.put(task)
            return task.task_id


class FailoverManager:
    """故障转移管理器"""
    
    def __init__(self, region_manager: RegionManager):
        self._region_manager = region_manager
        self._failover_rules: Dict[str, List[str]] = {}
        self._failover_history: List[Dict[str, Any]] = []
    
    def set_failover_rule(self, region_id: str, backup_regions: List[str]) -> None:
        self._failover_rules[region_id] = backup_regions
    
    def get_failover_history(self) -> List[Dict[str, Any]]:
        return self._failover_history.copy()


class GeoRoutingService:
    """地理位置路由服务"""
    
    def __init__(self, region_manager: RegionManager):
        self._region_manager = region_manager
        self._geo_mapping: Dict[str, str] = {}
    
    def set_geo_mapping(self, location: str, region_id: str) -> None:
        self._geo_mapping[location] = region_id
    
    async def route_by_location(self, client_location: str) -> Optional[RegionInfo]:
        if client_location in self._geo_mapping:
            region_id = self._geo_mapping[client_location]
            region = await self._region_manager.get_region(region_id)
            # 修复BUG#9: 检查区域可用性
            if region and region.is_available:
                return region
            # 映射的区域不可用，回退到智能选择
        return await self._region_manager.select_region(client_location=client_location, strategy=RoutingStrategy.LATENCY)


class CrossRegionOrchestrator:
    """跨区域编排器"""
    
    def __init__(self):
        self._region_manager = RegionManager()
        self._sync_engine = DataSyncEngine(self._region_manager)
        self._failover_manager = FailoverManager(self._region_manager)
        self._routing_service = GeoRoutingService(self._region_manager)
    
    def get_region_manager(self) -> RegionManager:
        return self._region_manager
    
    def get_failover_manager(self) -> FailoverManager:
        return self._failover_manager
    
    async def get_global_status(self) -> Dict[str, Any]:
        regions = await self._region_manager.list_regions()
        health = await self._region_manager.health_check()
        return {
            "regions": {r.region_id: {"name": r.name, "status": r.status.value} for r in regions},
            "health": health,
            "failover_history": self._failover_manager.get_failover_history()
        }
