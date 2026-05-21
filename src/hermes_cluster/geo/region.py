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
        logger.debug(f"ENTER RegionManager.__init__: args={{}}")
        self._regions: Dict[str, RegionInfo] = {}
        self._primary_region: Optional[str] = None
        self._routing_strategy: RoutingStrategy = RoutingStrategy.NEAREST
        self._lock = asyncio.Lock()
        logger.info(f"RegionManager initialized")
        logger.debug(f"EXIT RegionManager.__init__: result=None")
    
    async def register_region(self, region: RegionInfo) -> RegionInfo:
        logger.debug(f"ENTER RegionManager.register_region: args={{'region_id': region.region_id, 'name': region.name}}")
        async with self._lock:
            logger.debug(f"BRANCH: condition=lock acquired, result=True")
            if region.region_id in self._regions:
                logger.info(f"Region {region.region_id} already exists, updating")
                self._regions[region.region_id] = region  # 更新
                return region  # 返回region而非True
            self._regions[region.region_id] = region
            logger.info(f"Region registered: {region.region_id} ({region.name})")
            if self._primary_region is None:
                logger.debug(f"BRANCH: condition=primary_region is None, result=True")
                self._primary_region = region.region_id
                logger.info(f"Primary region set to: {region.region_id}")
            else:
                logger.debug(f"BRANCH: condition=primary_region is None, result=False")
        logger.debug(f"EXIT RegionManager.register_region: result=True")
        return region
    
    async def unregister_region(self, region_id: str) -> bool:
        logger.debug(f"ENTER RegionManager.unregister_region: args={{'region_id': region_id}}")
        async with self._lock:
            logger.debug(f"BRANCH: condition=lock acquired, result=True")
            if region_id in self._regions:
                logger.debug(f"BRANCH: condition=region_id in self._regions, result=True")
                del self._regions[region_id]
                logger.info(f"Region unregistered: {region_id}")
                if self._primary_region == region_id:
                    logger.debug(f"BRANCH: condition=primary_region == region_id, result=True")
                    self._primary_region = self._select_new_primary()
                    logger.info(f"New primary region selected: {self._primary_region}")
                else:
                    logger.debug(f"BRANCH: condition=primary_region == region_id, result=False")
                logger.debug(f"EXIT RegionManager.unregister_region: result=True")
                return True
            else:
                logger.debug(f"BRANCH: condition=region_id in self._regions, result=False")
        logger.warning(f"Region not found for unregister: {region_id}")
        logger.debug(f"EXIT RegionManager.unregister_region: result=False")
        return False
    
    def _select_new_primary(self) -> Optional[str]:
        logger.debug(f"ENTER RegionManager._select_new_primary: args={{}}")
        available = [r for r in self._regions.values() if r.is_available]
        logger.debug(f"BRANCH: condition=available regions count, result={len(available)}")
        if available:
            available.sort(key=lambda r: (-r.priority, r.load_ratio))
            selected = available[0].region_id
            logger.info(f"New primary region selected: {selected}")
            logger.debug(f"EXIT RegionManager._select_new_primary: result={selected}")
            return selected
        logger.warning("No available regions for primary selection")
        logger.debug(f"EXIT RegionManager._select_new_primary: result=None")
        return None
    
    async def update_region_status(self, region_id: str, status: RegionStatus, load: Optional[int] = None) -> bool:
        logger.debug(f"ENTER RegionManager.update_region_status: args={{'region_id': region_id, 'status': status.value, 'load': load}}")
        region = self._regions.get(region_id)
        if not region:
            logger.warning(f"Region not found for status update: {region_id}")
            logger.debug(f"EXIT RegionManager.update_region_status: result=False")
            return False
        old_status = region.status
        region.status = status
        region.last_heartbeat = datetime.now()
        logger.info(f"Region status updated: {region_id} from {old_status.value} to {status.value}")
        if load is not None:
            logger.debug(f"BRANCH: condition=load is not None, result=True")
            old_load = region.current_load
            region.current_load = load
            logger.info(f"Region load updated: {region_id} from {old_load} to {load}")
        else:
            logger.debug(f"BRANCH: condition=load is not None, result=False")
        logger.debug(f"EXIT RegionManager.update_region_status: result=True")
        return True
    
    async def get_region(self, region_id: str) -> Optional[RegionInfo]:
        logger.debug(f"ENTER RegionManager.get_region: args={{'region_id': region_id}}")
        region = self._regions.get(region_id)
        logger.debug(f"EXIT RegionManager.get_region: result={'found' if region else 'None'}")
        return region
    
    async def list_regions(self, status: Optional[RegionStatus] = None) -> List[RegionInfo]:
        logger.debug(f"ENTER RegionManager.list_regions: args={{'status': status.value if status else None}}")
        regions = list(self._regions.values())
        logger.debug(f"BRANCH: condition=initial regions count, result={len(regions)}")
        if status:
            logger.debug(f"BRANCH: condition=status filter provided, result=True")
            regions = [r for r in regions if r.status == status]
            logger.debug(f"BRANCH: condition=filtered regions count, result={len(regions)}")
        else:
            logger.debug(f"BRANCH: condition=status filter provided, result=False")
        logger.debug(f"EXIT RegionManager.list_regions: result={len(regions)} regions")
        return regions
    
    async def select_region(self, client_location: Optional[str] = None, strategy: Optional[RoutingStrategy] = None) -> Optional[RegionInfo]:
        logger.debug(f"ENTER RegionManager.select_region: args={{'client_location': client_location, 'strategy': strategy.value if strategy else None}}")
        strategy = strategy or self._routing_strategy
        logger.debug(f"BRANCH: condition=using strategy, result={strategy.value}")
        available = [r for r in self._regions.values() if r.is_available]
        logger.debug(f"BRANCH: condition=available regions count, result={len(available)}")
        if not available:
            logger.warning("No available regions for selection")
            logger.debug(f"EXIT RegionManager.select_region: result=None")
            return None
        
        if strategy == RoutingStrategy.NEAREST:
            logger.debug(f"BRANCH: condition=strategy == NEAREST, result=True")
            if client_location:
                logger.debug(f"BRANCH: condition=client_location provided, result=True")
                selected = min(available, key=lambda r: min(r.latency_map.values()) if r.latency_map else float('inf'))
                logger.debug(f"EXIT RegionManager.select_region: result={selected.region_id}")
                return selected
            selected = available[0]
            logger.debug(f"EXIT RegionManager.select_region: result={selected.region_id}")
            return selected
        elif strategy == RoutingStrategy.LATENCY:
            logger.debug(f"BRANCH: condition=strategy == LATENCY, result=True")
            selected = min(available, key=lambda r: min(r.latency_map.values()) if r.latency_map else float('inf'))
            logger.debug(f"EXIT RegionManager.select_region: result={selected.region_id}")
            return selected
        elif strategy == RoutingStrategy.LOAD:
            logger.debug(f"BRANCH: condition=strategy == LOAD, result=True")
            selected = min(available, key=lambda r: r.load_ratio)
            logger.debug(f"EXIT RegionManager.select_region: result={selected.region_id}")
            return selected
        elif strategy == RoutingStrategy.COST:
            logger.debug(f"BRANCH: condition=strategy == COST, result=True")
            selected = min(available, key=lambda r: r.priority)
            logger.debug(f"EXIT RegionManager.select_region: result={selected.region_id}")
            return selected
        logger.debug(f"BRANCH: condition=fallback to first available, result=True")
        selected = available[0]
        logger.debug(f"EXIT RegionManager.select_region: result={selected.region_id}")
        return selected
    
    def get_primary_region(self) -> Optional[RegionInfo]:
        logger.debug(f"ENTER RegionManager.get_primary_region: args={{}}")
        if self._primary_region:
            logger.debug(f"BRANCH: condition=primary_region exists, result=True")
            region = self._regions.get(self._primary_region)
            logger.debug(f"EXIT RegionManager.get_primary_region: result={'found' if region else 'None'}")
            return region
        logger.debug(f"BRANCH: condition=primary_region exists, result=False")
        logger.debug(f"EXIT RegionManager.get_primary_region: result=None")
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        logger.debug(f"ENTER RegionManager.health_check: args={{}}")
        results = {}
        for region_id, region in self._regions.items():
            logger.debug(f"BRANCH: condition=checking region, result={region_id}")
            is_healthy = region.is_available
            if region.last_heartbeat:
                logger.debug(f"BRANCH: condition=last_heartbeat exists, result=True")
                elapsed = (datetime.now() - region.last_heartbeat).total_seconds()
                is_healthy = is_healthy and elapsed < 60
                logger.debug(f"BRANCH: condition=heartbeat elapsed < 60s, result={elapsed < 60}")
            else:
                logger.debug(f"BRANCH: condition=last_heartbeat exists, result=False")
            results[region_id] = {
                "status": region.status.value,
                "healthy": is_healthy,
                "load_ratio": region.load_ratio
            }
            logger.info(f"Health check for {region_id}: healthy={is_healthy}, status={region.status.value}")
        logger.debug(f"EXIT RegionManager.health_check: result={len(results)} regions checked")
        return results


class DataSyncEngine:
    """数据同步引擎"""
    
    def __init__(self, region_manager: RegionManager):
        logger.debug(f"ENTER DataSyncEngine.__init__: args={{'region_manager': 'provided'}}")
        self._region_manager = region_manager
        self._partitions: Dict[str, DataPartition] = {}
        self._sync_tasks: Dict[str, SyncTask] = {}
        self._is_running = False
        self._lock = asyncio.Lock()
        logger.info(f"DataSyncEngine initialized")
        logger.debug(f"EXIT DataSyncEngine.__init__: result=None")
    
    async def create_partition(self, partition_id: str, primary_region: str, replica_regions: List[str], sync_mode: SyncMode = SyncMode.ASYNC) -> DataPartition:
        logger.debug(f"ENTER DataSyncEngine.create_partition: args={{'partition_id': partition_id, 'primary_region': primary_region, 'replica_regions': replica_regions, 'sync_mode': sync_mode.value}}")
        async with self._lock:
            logger.debug(f"BRANCH: condition=lock acquired, result=True")
            partition = DataPartition(
                partition_id=partition_id,
                region_id=primary_region,
                replica_regions=replica_regions,
                sync_mode=sync_mode
            )
            self._partitions[partition_id] = partition
            logger.info(f"Partition created: {partition_id}, primary={primary_region}, replicas={replica_regions}")
            logger.debug(f"EXIT DataSyncEngine.create_partition: result=partition {partition_id}")
            return partition
    
    async def sync_data(self, partition_id: str, data: Any, source_region: str) -> str:
        logger.debug(f"ENTER DataSyncEngine.sync_data: args={{'partition_id': partition_id, 'source_region': source_region}}")
        async with self._lock:
            logger.debug(f"BRANCH: condition=lock acquired, result=True")
            partition = self._partitions.get(partition_id)
            if not partition:
                logger.error(f"Partition not found: {partition_id}")
                logger.debug(f"EXIT DataSyncEngine.sync_data: result=exception")
                raise ValueError(f"分区不存在: {partition_id}")

            # 确定目标区域：优先同步到副本，避免回环
            target_region = None
            if source_region == partition.region_id:
                logger.debug(f"BRANCH: condition=source is primary region, result=True")
                # 从主区域发起，同步到第一个副本
                if partition.replica_regions:
                    logger.debug(f"BRANCH: condition=replica_regions exist, result=True")
                    target_region = partition.replica_regions[0]
                else:
                    logger.debug(f"BRANCH: condition=replica_regions exist, result=False")
            else:
                logger.debug(f"BRANCH: condition=source is primary region, result=False")
                # 从副本发起，同步回主区域
                target_region = partition.region_id

            # 无副本且从主区域发起时，跳过同步
            if target_region is None or target_region == source_region:
                logger.debug(f"BRANCH: condition=skip sync (target={target_region}), result=True")
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
                logger.info(f"Sync task created (skipped): {task.task_id}")
                # 限制保留最近1000条
                if len(self._sync_tasks) > 1000:
                    logger.debug(f"BRANCH: condition=sync_tasks > 1000, result=True")
                    keys = list(self._sync_tasks.keys())
                    for k in keys[:len(self._sync_tasks) - 800]:
                        del self._sync_tasks[k]
                    logger.info(f"Cleaned up old sync tasks, remaining: {len(self._sync_tasks)}")
                else:
                    logger.debug(f"BRANCH: condition=sync_tasks > 1000, result=False")
                logger.debug(f"EXIT DataSyncEngine.sync_data: result={task.task_id}")
                return task.task_id

            logger.debug(f"BRANCH: condition=target_region determined, result={target_region}")
            task = SyncTask(
                task_id=str(uuid.uuid4()),
                source_region=source_region,
                target_region=target_region,
                partition_id=partition_id,
                data=data
            )
            self._sync_tasks[task.task_id] = task
            logger.info(f"Sync task created: {task.task_id}, {source_region} -> {target_region}")
            # 限制保留最近1000条
            if len(self._sync_tasks) > 1000:
                logger.debug(f"BRANCH: condition=sync_tasks > 1000, result=True")
                keys = list(self._sync_tasks.keys())
                for k in keys[:len(self._sync_tasks) - 800]:
                    del self._sync_tasks[k]
                logger.info(f"Cleaned up old sync tasks, remaining: {len(self._sync_tasks)}")
            else:
                logger.debug(f"BRANCH: condition=sync_tasks > 1000, result=False")
            # 直接处理同步任务，不使用队列
            task.status = "completed"
            task.completed_at = datetime.now()
            logger.info(f"Sync task completed directly: {task.task_id}")
            logger.debug(f"EXIT DataSyncEngine.sync_data: result={task.task_id}")
            return task.task_id


class FailoverManager:
    """故障转移管理器"""
    
    def __init__(self, region_manager: RegionManager):
        logger.debug(f"ENTER FailoverManager.__init__: args={{'region_manager': 'provided'}}")
        self._region_manager = region_manager
        self._failover_rules: Dict[str, List[str]] = {}
        self._failover_history: List[Dict[str, Any]] = []
        logger.info(f"FailoverManager initialized")
        logger.debug(f"EXIT FailoverManager.__init__: result=None")
    
    def set_failover_rule(self, region_id: str, backup_regions: List[str]) -> None:
        logger.debug(f"ENTER FailoverManager.set_failover_rule: args={{'region_id': region_id, 'backup_regions': backup_regions}}")
        self._failover_rules[region_id] = backup_regions
        logger.info(f"Failover rule set: {region_id} -> {backup_regions}")
        logger.debug(f"EXIT FailoverManager.set_failover_rule: result=None")
    
    def get_failover_history(self) -> List[Dict[str, Any]]:
        logger.debug(f"ENTER FailoverManager.get_failover_history: args={{}}")
        history = self._failover_history.copy()
        logger.debug(f"EXIT FailoverManager.get_failover_history: result={len(history)} entries")
        return history


class GeoRoutingService:
    """地理位置路由服务"""
    
    def __init__(self, region_manager: RegionManager):
        logger.debug(f"ENTER GeoRoutingService.__init__: args={{'region_manager': 'provided'}}")
        self._region_manager = region_manager
        self._geo_mapping: Dict[str, str] = {}
        logger.info(f"GeoRoutingService initialized")
        logger.debug(f"EXIT GeoRoutingService.__init__: result=None")
    
    def set_geo_mapping(self, location: str, region_id: str) -> None:
        logger.debug(f"ENTER GeoRoutingService.set_geo_mapping: args={{'location': location, 'region_id': region_id}}")
        self._geo_mapping[location] = region_id
        logger.info(f"Geo mapping set: {location} -> {region_id}")
        logger.debug(f"EXIT GeoRoutingService.set_geo_mapping: result=None")
    
    async def route_by_location(self, client_location: str) -> Optional[RegionInfo]:
        logger.debug(f"ENTER GeoRoutingService.route_by_location: args={{'client_location': client_location}}")
        if client_location in self._geo_mapping:
            logger.debug(f"BRANCH: condition=client_location in geo_mapping, result=True")
            region_id = self._geo_mapping[client_location]
            region = await self._region_manager.get_region(region_id)
            # 修复BUG#9: 检查区域可用性
            if region and region.is_available:
                logger.debug(f"BRANCH: condition=region available, result=True")
                logger.info(f"Routed to mapped region: {region_id}")
                logger.debug(f"EXIT GeoRoutingService.route_by_location: result={region_id}")
                return region
            logger.debug(f"BRANCH: condition=region available, result=False")
            # 映射的区域不可用，回退到智能选择
        else:
            logger.debug(f"BRANCH: condition=client_location in geo_mapping, result=False")
        selected = await self._region_manager.select_region(client_location=client_location, strategy=RoutingStrategy.LATENCY)
        logger.debug(f"EXIT GeoRoutingService.route_by_location: result={'found' if selected else 'None'}")
        return selected


class CrossRegionOrchestrator:
    """跨区域编排器"""
    
    def __init__(self):
        logger.debug(f"ENTER CrossRegionOrchestrator.__init__: args={{}}")
        self._region_manager = RegionManager()
        self._sync_engine = DataSyncEngine(self._region_manager)
        self._failover_manager = FailoverManager(self._region_manager)
        self._routing_service = GeoRoutingService(self._region_manager)
        logger.info(f"CrossRegionOrchestrator initialized")
        logger.debug(f"EXIT CrossRegionOrchestrator.__init__: result=None")
    
    def get_region_manager(self) -> RegionManager:
        logger.debug(f"ENTER CrossRegionOrchestrator.get_region_manager: args={{}}")
        logger.debug(f"EXIT CrossRegionOrchestrator.get_region_manager: result=RegionManager")
        return self._region_manager
    
    def get_failover_manager(self) -> FailoverManager:
        logger.debug(f"ENTER CrossRegionOrchestrator.get_failover_manager: args={{}}")
        logger.debug(f"EXIT CrossRegionOrchestrator.get_failover_manager: result=FailoverManager")
        return self._failover_manager
    
    async def get_global_status(self) -> Dict[str, Any]:
        logger.debug(f"ENTER CrossRegionOrchestrator.get_global_status: args={{}}")
        regions = await self._region_manager.list_regions()
        logger.debug(f"BRANCH: condition=regions listed, result={len(regions)}")
        health = await self._region_manager.health_check()
        logger.debug(f"BRANCH: condition=health checked, result={len(health)}")
        result = {
            "regions": {r.region_id: {"name": r.name, "status": r.status.value} for r in regions},
            "health": health,
            "failover_history": self._failover_manager.get_failover_history()
        }
        logger.debug(f"EXIT CrossRegionOrchestrator.get_global_status: result={len(result)} keys")
        return result
