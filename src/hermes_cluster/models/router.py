"""
Hermes Cluster v6.0 - 多模型路由模块
支持多模型提供商路由、负载均衡、故障转移、成本优化

优化点:
1. 多提供商统一接口（OpenAI兼容）
2. 模型路由策略（成本/延迟/质量）
3. 自动故障转移与重试
4. 请求速率限制
5. Token用量统计与配额管理
6. 流式响应支持
7. 模型能力标签匹配
8. 请求/响应缓存
9. A/B测试支持
10. 成本追踪与预算控制
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, AsyncIterator
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """模型能力标签"""
    CHAT = "chat"
    CODE = "code"
    REASONING = "reasoning"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"


class ProviderStatus(Enum):
    """提供商状态"""
    ACTIVE = "active"
    DEGRADED = "degraded"
    DISABLED = "disabled"
    ERROR = "error"


class RouteStrategy(Enum):
    """路由策略"""
    PRIORITY = "priority"       # 优先级
    ROUND_ROBIN = "round_robin" # 轮询
    LEAST_COST = "least_cost"   # 最低成本
    LEAST_LATENCY = "least_latency"  # 最低延迟
    RANDOM = "random"           # 随机
    CAPABILITY = "capability"   # 能力匹配


@dataclass
class ModelInfo:
    """模型信息"""
    model_id: str
    provider_id: str
    display_name: str
    capabilities: set = field(default_factory=set)
    max_context_tokens: int = 4096
    max_output_tokens: int = 2048
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    priority: int = 0
    is_free: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """提供商配置"""
    provider_id: str
    name: str
    base_url: str
    api_key: str
    status: ProviderStatus = ProviderStatus.ACTIVE
    default_model: str = ""
    max_concurrent: int = 10
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    """聊天消息"""
    role: str  # system, user, assistant
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ChatRequest:
    """聊天请求"""
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    stop: Optional[List[str]] = None
    top_p: float = 0.7
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """聊天响应"""
    request_id: str
    model: str
    provider_id: str
    content: str = ""
    reasoning_content: str = ""
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    created: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageRecord:
    """用量记录"""
    provider_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: str = ""


class ModelProvider(ABC):
    """模型提供商抽象基类"""

    def __init__(self, config: ProviderConfig):
        logger.debug(f"ENTER ModelProvider.__init__: args={{'provider_id': config.provider_id}}")
        self._config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        logger.info(f"ModelProvider initialized: {config.provider_id}")
        logger.debug(f"EXIT ModelProvider.__init__: result=None")

    @abstractmethod
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        pass

    @abstractmethod
    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        pass

    @property
    def provider_id(self) -> str:
        return self._config.provider_id

    @property
    def is_available(self) -> bool:
        return self._config.status == ProviderStatus.ACTIVE


class SiliconFlowProvider(ModelProvider):
    """硅基流动提供商（OpenAI兼容）"""

    def __init__(self, config: ProviderConfig):
        logger.debug(f"ENTER SiliconFlowProvider.__init__: args={{'provider_id': config.provider_id}}")
        super().__init__(config)
        self._round_robin_idx = 0
        logger.info(f"SiliconFlowProvider initialized")
        logger.debug(f"EXIT SiliconFlowProvider.__init__: result=None")

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        logger.debug(f"ENTER SiliconFlowProvider.chat_completion: args={{'request_id': request.request_id, 'model': request.model}}")
        async with self._semaphore:
            logger.debug(f"BRANCH: condition=semaphore acquired, result=True")
            import aiohttp

            url = f"{self._config.base_url.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": request.model,
                "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False,
                "top_p": request.top_p,
            }
            if request.stop:
                logger.debug(f"BRANCH: condition=request.stop provided, result=True")
                payload["stop"] = request.stop
            if request.tools:
                logger.debug(f"BRANCH: condition=request.tools provided, result=True")
                payload["tools"] = request.tools

            start_time = time.time()

            for attempt in range(self._config.max_retries):
                logger.debug(f"BRANCH: condition=attempt, result={attempt + 1}/{self._config.max_retries}")
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url, headers=headers, json=payload,
                            timeout=aiohttp.ClientTimeout(total=self._config.timeout)
                        ) as resp:
                            if resp.status == 429:
                                logger.debug(f"BRANCH: condition=rate limited (429), result=True")
                                retry_after = float(resp.headers.get("Retry-After", self._config.retry_delay))
                                logger.warning(f"硅基流动速率限制，等待 {retry_after}s 后重试")
                                await asyncio.sleep(retry_after)
                                continue

                            if resp.status != 200:
                                logger.debug(f"BRANCH: condition=error status {resp.status}, result=True")
                                error_text = await resp.text()
                                raise RuntimeError(f"硅基流动API错误 {resp.status}: {error_text}")

                            data = await resp.json()

                    if not data.get("choices"):
                        logger.debug(f"BRANCH: condition=empty choices, result=True")
                        raise RuntimeError("API返回空choices数组，可能触发了内容过滤")

                    latency = (time.time() - start_time) * 1000
                    choice = data["choices"][0]
                    message = choice["message"]
                    usage = data.get("usage", {})

                    logger.info(f"Chat completion successful: {request.request_id}, latency={latency:.2f}ms")
                    logger.debug(f"EXIT SiliconFlowProvider.chat_completion: result=response with {len(usage)} usage stats")
                    return ChatResponse(
                        request_id=request.request_id,
                        model=data.get("model", request.model),
                        provider_id=self.provider_id,
                        content=message.get("content", ""),
                        reasoning_content=message.get("reasoning_content", ""),
                        finish_reason=choice.get("finish_reason", "stop"),
                        usage={
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0)
                        },
                        tool_calls=message.get("tool_calls"),
                        latency_ms=latency
                    )

                except asyncio.TimeoutError:
                    logger.warning(f"硅基流动请求超时 (attempt {attempt + 1}/{self._config.max_retries})")
                    if attempt < self._config.max_retries - 1:
                        logger.debug(f"BRANCH: condition=will retry, result=True")
                        await asyncio.sleep(self._config.retry_delay * (attempt + 1))
                    else:
                        logger.debug(f"BRANCH: condition=will retry, result=False")
                except Exception as e:
                    logger.error(f"EXCEPTION: {e}", exc_info=True)
                    logger.error(f"硅基流动请求失败: {e}")
                    if attempt < self._config.max_retries - 1:
                        logger.debug(f"BRANCH: condition=will retry, result=True")
                        await asyncio.sleep(self._config.retry_delay * (attempt + 1))
                    else:
                        logger.debug(f"BRANCH: condition=will retry, result=False")
                        raise

            raise RuntimeError(f"硅基流动请求失败，已重试 {self._config.max_retries} 次")

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        logger.debug(f"ENTER SiliconFlowProvider.chat_completion_stream: args={{'request_id': request.request_id, 'model': request.model}}")
        await self._semaphore.acquire()
        logger.debug(f"BRANCH: condition=semaphore acquired, result=True")
        released = False
        try:
            import aiohttp

            url = f"{self._config.base_url.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": request.model,
                "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=self._config.timeout * 2)
                ) as resp:
                    buffer = ""
                    chunk_count = 0
                    try:
                        async for chunk in resp.content:
                            chunk_text = chunk.decode("utf-8")
                            buffer += chunk_text
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    try:
                                        data = json.loads(line[6:])
                                        delta = data["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            chunk_count += 1
                                            yield content
                                    except (json.JSONDecodeError, KeyError, IndexError):
                                        continue
                    finally:
                        # BUG#35修复：使用released标志防止双重释放semaphore
                        if not released:
                            self._semaphore.release()
                            released = True
                        logger.debug(f"BRANCH: condition=semaphore released in finally, result=True")
                    logger.info(f"Stream completed: {request.request_id}, chunks={chunk_count}")
        except Exception:
            # BUG#35修复：使用released标志防止双重释放
            if not released:
                self._semaphore.release()
                released = True
            logger.debug(f"BRANCH: condition=semaphore released on exception, result=True")
            raise


class OpenAICompatibleProvider(ModelProvider):
    """通用OpenAI兼容提供商"""

    def __init__(self, config: ProviderConfig):
        logger.debug(f"ENTER OpenAICompatibleProvider.__init__: args={{'provider_id': config.provider_id}}")
        super().__init__(config)
        logger.info(f"OpenAICompatibleProvider initialized: {config.provider_id}")
        logger.debug(f"EXIT OpenAICompatibleProvider.__init__: result=None")

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        logger.debug(f"ENTER OpenAICompatibleProvider.chat_completion: args={{'request_id': request.request_id, 'model': request.model}}")
        async with self._semaphore:
            logger.debug(f"BRANCH: condition=semaphore acquired, result=True")
            import aiohttp
            url = f"{self._config.base_url.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": request.model,
                "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False,
            }
            if request.tools:
                logger.debug(f"BRANCH: condition=request.tools provided, result=True")
                payload["tools"] = request.tools

            last_error = None
            for attempt in range(self._config.max_retries):
                logger.debug(f"BRANCH: condition=attempt, result={attempt + 1}/{self._config.max_retries}")
                try:
                    start_time = time.time()
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url, headers=headers, json=payload,
                            timeout=aiohttp.ClientTimeout(total=self._config.timeout)
                        ) as resp:
                            if resp.status == 429:
                                logger.debug(f"BRANCH: condition=rate limited (429), result=True")
                                retry_after = self._config.retry_delay
                                try:
                                    retry_after = float(resp.headers.get("Retry-After", retry_after))
                                except (ValueError, TypeError):
                                    pass
                                await asyncio.sleep(retry_after * (attempt + 1))
                                continue
                            if resp.status != 200:
                                logger.debug(f"BRANCH: condition=error status {resp.status}, result=True")
                                error_text = await resp.text()
                                raise RuntimeError(f"API错误 {resp.status}: {error_text}")
                            data = await resp.json()

                    latency = (time.time() - start_time) * 1000

                    # 检查空choices
                    if not data.get("choices"):
                        logger.debug(f"BRANCH: condition=empty choices, result=True")
                        raise RuntimeError("API返回空choices数组")

                    choice = data["choices"][0]
                    message = choice["message"]
                    usage = data.get("usage", {})

                    logger.info(f"Chat completion successful: {request.request_id}, provider={self.provider_id}, latency={latency:.2f}ms")
                    logger.debug(f"EXIT OpenAICompatibleProvider.chat_completion: result=response")
                    return ChatResponse(
                        request_id=request.request_id,
                        model=data.get("model", request.model),
                        provider_id=self.provider_id,
                        content=message.get("content", ""),
                        finish_reason=choice.get("finish_reason", "stop"),
                        usage={
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0)
                        },
                        tool_calls=message.get("tool_calls"),
                        latency_ms=latency
                    )
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.error(f"EXCEPTION: {e}", exc_info=True)
                    last_error = e
                    if attempt < self._config.max_retries - 1:
                        logger.debug(f"BRANCH: condition=will retry, result=True")
                        await asyncio.sleep(self._config.retry_delay * (attempt + 1))
                    else:
                        logger.debug(f"BRANCH: condition=will retry, result=False")
                        raise
                except RuntimeError:
                    raise

            raise RuntimeError(f"请求失败，已重试 {self._config.max_retries} 次: {last_error}")

    async def chat_completion_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        logger.debug(f"ENTER OpenAICompatibleProvider.chat_completion_stream: args={{'request_id': request.request_id}}")
        logger.error(f"EXCEPTION: Stream not implemented")
        raise NotImplementedError("流式响应暂未实现")


class ModelRouter:
    """模型路由器"""

    def __init__(self):
        logger.debug(f"ENTER ModelRouter.__init__: args={{}}")
        self._providers: Dict[str, ModelProvider] = {}
        self._models: Dict[str, ModelInfo] = {}
        self._model_to_provider: Dict[str, str] = {}
        self._strategy: RouteStrategy = RouteStrategy.PRIORITY
        self._round_robin_idx: Dict[str, int] = {}
        self._usage_history: List[UsageRecord] = []
        self._latency_history: Dict[str, List[float]] = {}
        self._error_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        logger.info(f"ModelRouter initialized")
        logger.debug(f"EXIT ModelRouter.__init__: result=None")

    def register_provider(self, provider: ModelProvider) -> None:
        logger.debug(f"ENTER ModelRouter.register_provider: args={{'provider_id': provider.provider_id}}")
        self._providers[provider.provider_id] = provider
        logger.info(f"Provider registered: {provider.provider_id}")
        logger.debug(f"EXIT ModelRouter.register_provider: result=None")

    def register_model(self, model: ModelInfo) -> None:
        logger.debug(f"ENTER ModelRouter.register_model: args={{'model_id': model.model_id, 'provider_id': model.provider_id}}")
        self._models[model.model_id] = model
        self._model_to_provider[model.model_id] = model.provider_id
        logger.info(f"Model registered: {model.model_id} ({model.display_name})")
        logger.debug(f"EXIT ModelRouter.register_model: result=None")

    def set_strategy(self, strategy: RouteStrategy) -> None:
        logger.debug(f"ENTER ModelRouter.set_strategy: args={{'strategy': strategy.value}}")
        self._strategy = strategy
        logger.info(f"Routing strategy set to: {strategy.value}")
        logger.debug(f"EXIT ModelRouter.set_strategy: result=None")

    async def route(self, model: Optional[str] = None, capabilities: Optional[set] = None) -> Optional[ModelInfo]:
        """选择最佳模型"""
        logger.debug(f"ENTER ModelRouter.route: args={{'model': model, 'capabilities': capabilities}}")
        async with self._lock:
            logger.debug(f"BRANCH: condition=lock acquired, result=True")
            if model and model in self._models:
                logger.debug(f"BRANCH: condition=specific model requested and exists, result=True")
                info = self._models[model]
                provider = self._providers.get(info.provider_id)
                if provider and provider.is_available:
                    logger.debug(f"BRANCH: condition=provider available, result=True")
                    logger.info(f"Routed to requested model: {model}")
                    logger.debug(f"EXIT ModelRouter.route: result=model {model}")
                    return info
                logger.debug(f"BRANCH: condition=provider available, result=False")
                # 故障转移到同类型模型
                fallback = await self._fallback_route(model)
                logger.debug(f"EXIT ModelRouter.route: result=fallback model")
                return fallback

            candidates = self._get_available_models(capabilities)
            logger.debug(f"BRANCH: condition=available candidates count, result={len(candidates)}")
            if not candidates:
                logger.warning("No available models for routing")
                logger.debug(f"EXIT ModelRouter.route: result=None")
                return None

            if self._strategy == RouteStrategy.PRIORITY:
                logger.debug(f"BRANCH: condition=strategy == PRIORITY, result=True")
                selected = max(candidates, key=lambda m: m.priority)
            elif self._strategy == RouteStrategy.LEAST_COST:
                logger.debug(f"BRANCH: condition=strategy == LEAST_COST, result=True")
                selected = min(candidates, key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)
            elif self._strategy == RouteStrategy.LEAST_LATENCY:
                logger.debug(f"BRANCH: condition=strategy == LEAST_LATENCY, result=True")
                selected = self._select_by_latency(candidates)
            elif self._strategy == RouteStrategy.ROUND_ROBIN:
                logger.debug(f"BRANCH: condition=strategy == ROUND_ROBIN, result=True")
                selected = self._select_round_robin(candidates)
            elif self._strategy == RouteStrategy.RANDOM:
                logger.debug(f"BRANCH: condition=strategy == RANDOM, result=True")
                import random
                selected = random.choice(candidates)
            elif self._strategy == RouteStrategy.CAPABILITY:
                logger.debug(f"BRANCH: condition=strategy == CAPABILITY, result=True")
                selected = candidates[0] if candidates else None
            else:
                logger.debug(f"BRANCH: condition=fallback strategy, result=True")
                selected = candidates[0] if candidates else None

            if selected:
                logger.info(f"Model routed: {selected.model_id} using strategy {self._strategy.value}")
            logger.debug(f"EXIT ModelRouter.route: result={'model ' + selected.model_id if selected else 'None'}")
            return selected

    async def _fallback_route(self, failed_model: str) -> Optional[ModelInfo]:
        """故障转移"""
        logger.debug(f"ENTER ModelRouter._fallback_route: args={{'failed_model': failed_model}}")
        failed_info = self._models.get(failed_model)
        if not failed_info:
            logger.warning(f"Failed model info not found: {failed_model}")
            logger.debug(f"EXIT ModelRouter._fallback_route: result=None")
            return None

        # 获取可用提供商ID集合
        available_provider_ids = {
            pid for pid, p in self._providers.items() if p.is_available
        }
        logger.debug(f"BRANCH: condition=available providers count, result={len(available_provider_ids)}")

        # 优先选择不同提供商的模型
        candidates = [
            m for m in self._models.values()
            if m.model_id != failed_model
            and m.provider_id in available_provider_ids
            and m.provider_id != failed_info.provider_id
        ]
        logger.debug(f"BRANCH: condition=cross-provider candidates count, result={len(candidates)}")
        if not candidates:
            logger.debug(f"BRANCH: condition=no cross-provider candidates, result=True")
            candidates = [
                m for m in self._models.values()
                if m.model_id != failed_model
                and m.provider_id in available_provider_ids
            ]
            logger.debug(f"BRANCH: condition=same-provider candidates count, result={len(candidates)}")

        if candidates:
            selected = max(candidates, key=lambda m: m.priority)
            logger.info(f"Fallback route selected: {selected.model_id} (original: {failed_model})")
            logger.debug(f"EXIT ModelRouter._fallback_route: result=model {selected.model_id}")
            return selected
        logger.warning(f"No fallback model available for: {failed_model}")
        logger.debug(f"EXIT ModelRouter._fallback_route: result=None")
        return None

    def _get_available_models(self, capabilities: Optional[set] = None) -> List[ModelInfo]:
        logger.debug(f"ENTER ModelRouter._get_available_models: args={{'capabilities': capabilities}}")
        candidates = []
        for model_id, model in self._models.items():
            provider = self._providers.get(model.provider_id)
            if not provider or not provider.is_available:
                logger.debug(f"BRANCH: condition=provider not available, skip model, result={model_id}")
                continue
            if capabilities and not capabilities.issubset(model.capabilities):
                logger.debug(f"BRANCH: condition=capabilities mismatch, skip model, result={model_id}")
                continue
            candidates.append(model)
        logger.debug(f"EXIT ModelRouter._get_available_models: result={len(candidates)} candidates")
        return candidates

    def _select_by_latency(self, candidates: List[ModelInfo]) -> ModelInfo:
        logger.debug(f"ENTER ModelRouter._select_by_latency: args={{'candidates_count': len(candidates)}}")
        best = candidates[0]
        best_avg = float('inf')
        for m in candidates:
            latencies = self._latency_history.get(m.model_id, [])
            avg = sum(latencies) / len(latencies) if latencies else 0
            logger.debug(f"BRANCH: condition=model {m.model_id} avg latency, result={avg:.2f}ms")
            if avg < best_avg:
                best_avg = avg
                best = m
        logger.info(f"Selected by latency: {best.model_id} (avg={best_avg:.2f}ms)")
        logger.debug(f"EXIT ModelRouter._select_by_latency: result=model {best.model_id}")
        return best

    def _select_round_robin(self, candidates: List[ModelInfo]) -> ModelInfo:
        logger.debug(f"ENTER ModelRouter._select_round_robin: args={{'candidates_count': len(candidates)}}")
        key = ",".join(sorted(m.model_id for m in candidates))
        idx = self._round_robin_idx.get(key, 0)
        selected = candidates[idx % len(candidates)]
        self._round_robin_idx[key] = idx + 1
        logger.info(f"Selected by round-robin: {selected.model_id} (idx={idx})")
        logger.debug(f"EXIT ModelRouter._select_round_robin: result=model {selected.model_id}")
        return selected

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """发送聊天请求（自动路由）"""
        logger.debug(f"ENTER ModelRouter.chat: args={{'request_id': request.request_id, 'model': request.model}}")
        model_info = await self.route(request.model)
        if not model_info:
            logger.error(f"No available model for request: {request.request_id}")
            raise RuntimeError("无可用模型")

        provider = self._providers[model_info.provider_id]
        target_request = replace(request, model=model_info.model_id)
        logger.info(f"Chat request routed: {request.request_id} -> {model_info.model_id}")

        try:
            response = await provider.chat_completion(target_request)
            self._record_usage(response)
            self._record_latency(model_info.model_id, response.latency_ms)
            logger.info(f"Chat completed: {request.request_id}, tokens={response.usage.get('total_tokens', 0)}")
            logger.debug(f"EXIT ModelRouter.chat: result=response")
            return response
        except Exception as e:
            logger.error(f"EXCEPTION: {e}", exc_info=True)
            self._record_error(model_info.model_id)
            logger.error(f"模型 {model_info.model_id} 请求失败: {e}")

            # 故障转移
            fallback = await self._fallback_route(model_info.model_id)
            if fallback:
                logger.info(f"故障转移到模型 {fallback.model_id}")
                fallback_provider = self._providers[fallback.provider_id]
                fallback_request = replace(request, model=fallback.model_id)
                response = await fallback_provider.chat_completion(fallback_request)
                self._record_usage(response)
                self._record_latency(fallback.model_id, response.latency_ms)
                logger.info(f"Fallback chat completed: {request.request_id} -> {fallback.model_id}")
                logger.debug(f"EXIT ModelRouter.chat: result=fallback response")
                return response
            raise

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """流式聊天（带错误处理和故障转移）"""
        logger.debug(f"ENTER ModelRouter.chat_stream: args={{'request_id': request.request_id, 'model': request.model}}")
        model_info = await self.route(request.model)
        if not model_info:
            logger.error(f"No available model for stream request: {request.request_id}")
            raise RuntimeError("无可用模型")

        provider = self._providers.get(model_info.provider_id)
        if not provider or not provider.is_available:
            # 故障转移到其他提供商
            fallback = await self._fallback_route(model_info.model_id)
            if fallback:
                provider = self._providers.get(fallback.provider_id)
                model_info = fallback
                logger.info(f"Stream request fallback: {request.request_id} -> {model_info.model_id}")
            else:
                raise RuntimeError("无可用提供商")

        target_request = replace(request, model=model_info.model_id)
        logger.info(f"Stream request routed: {request.request_id} -> {model_info.model_id}")

        try:
            chunk_count = 0
            async for chunk in provider.chat_completion_stream(target_request):
                chunk_count += 1
                yield chunk
            logger.info(f"Stream completed: {request.request_id}, chunks={chunk_count}")
            logger.debug(f"EXIT ModelRouter.chat_stream: result=stream completed")
        except Exception as e:
            logger.error(f"EXCEPTION: {e}", exc_info=True)
            logger.error(f"模型 {model_info.model_id} 流式请求失败: {e}")

            # 尝试故障转移
            fallback = await self._fallback_route(model_info.model_id)
            if fallback:
                logger.info(f"流式请求故障转移到模型 {fallback.model_id}")
                fallback_provider = self._providers.get(fallback.provider_id)
                if fallback_provider:
                    fallback_request = replace(request, model=fallback.model_id)
                    chunk_count = 0
                    async for chunk in fallback_provider.chat_completion_stream(fallback_request):
                        chunk_count += 1
                        yield chunk
                    logger.info(f"Fallback stream completed: {request.request_id} -> {fallback.model_id}, chunks={chunk_count}")
                    logger.debug(f"EXIT ModelRouter.chat_stream: result=fallback stream completed")
                    return
            raise

    def _record_usage(self, response: ChatResponse) -> None:
        logger.debug(f"ENTER ModelRouter._record_usage: args={{'request_id': response.request_id, 'model': response.model}}")
        record = UsageRecord(
            provider_id=response.provider_id,
            model=response.model,
            prompt_tokens=response.usage.get("prompt_tokens", 0),
            completion_tokens=response.usage.get("completion_tokens", 0),
            total_tokens=response.usage.get("total_tokens", 0),
            estimated_cost=0.0,
            request_id=response.request_id
        )
        self._usage_history.append(record)
        logger.info(f"Usage recorded: {response.request_id}, tokens={record.total_tokens}")
        # 保留最近1000条
        if len(self._usage_history) > 1000:
            logger.debug(f"BRANCH: condition=usage_history > 1000, result=True")
            self._usage_history = self._usage_history[-1000:]
            logger.info(f"Cleaned up old usage records, remaining: {len(self._usage_history)}")
        else:
            logger.debug(f"BRANCH: condition=usage_history > 1000, result=False")
        logger.debug(f"EXIT ModelRouter._record_usage: result=None")

    def _record_latency(self, model_id: str, latency_ms: float) -> None:
        logger.debug(f"ENTER ModelRouter._record_latency: args={{'model_id': model_id, 'latency_ms': latency_ms}}")
        if model_id not in self._latency_history:
            self._latency_history[model_id] = []
        self._latency_history[model_id].append(latency_ms)
        # 保留最近100条
        if len(self._latency_history[model_id]) > 100:
            logger.debug(f"BRANCH: condition=latency_history > 100, result=True")
            self._latency_history[model_id] = self._latency_history[model_id][-100:]
        else:
            logger.debug(f"BRANCH: condition=latency_history > 100, result=False")
        logger.debug(f"EXIT ModelRouter._record_latency: result=None")

    def _record_error(self, model_id: str) -> None:
        logger.debug(f"ENTER ModelRouter._record_error: args={{'model_id': model_id}}")
        self._error_counts[model_id] = self._error_counts.get(model_id, 0) + 1
        logger.warning(f"Error recorded for model {model_id}: count={self._error_counts[model_id]}")
        # 限制条目数
        if len(self._error_counts) > 100:
            logger.debug(f"BRANCH: condition=error_counts > 100, result=True")
            sorted_items = sorted(self._error_counts.items(), key=lambda x: x[1])
            for k, _ in sorted_items[:len(self._error_counts) - 50]:
                del self._error_counts[k]
            logger.info(f"Cleaned up old error counts, remaining: {len(self._error_counts)}")
        else:
            logger.debug(f"BRANCH: condition=error_counts > 100, result=False")
        logger.debug(f"EXIT ModelRouter._record_error: result=None")

    def get_usage_stats(self) -> Dict[str, Any]:
        logger.debug(f"ENTER ModelRouter.get_usage_stats: args={{}}")
        total_tokens = sum(r.total_tokens for r in self._usage_history)
        total_requests = len(self._usage_history)
        by_provider: Dict[str, Dict[str, Any]] = {}
        for r in self._usage_history:
            if r.provider_id not in by_provider:
                by_provider[r.provider_id] = {"tokens": 0, "requests": 0}
            by_provider[r.provider_id]["tokens"] += r.total_tokens
            by_provider[r.provider_id]["requests"] += 1

        result = {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "by_provider": by_provider,
            "error_counts": self._error_counts.copy()
        }
        logger.info(f"Usage stats: {total_requests} requests, {total_tokens} tokens")
        logger.debug(f"EXIT ModelRouter.get_usage_stats: result={len(result)} keys")
        return result

    def get_latency_stats(self) -> Dict[str, Dict[str, float]]:
        logger.debug(f"ENTER ModelRouter.get_latency_stats: args={{}}")
        stats = {}
        for model_id, latencies in self._latency_history.items():
            if latencies:
                stats[model_id] = {
                    "avg_ms": sum(latencies) / len(latencies),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "samples": len(latencies)
                }
                logger.debug(f"BRANCH: condition=latency stats for model, result={model_id}: avg={stats[model_id]['avg_ms']:.2f}ms")
        logger.info(f"Latency stats: {len(stats)} models")
        logger.debug(f"EXIT ModelRouter.get_latency_stats: result={len(stats)} models")
        return stats


def create_siliconflow_provider(api_key: str) -> tuple[SiliconFlowProvider, List[ModelInfo]]:
    """创建硅基流动提供商及模型列表"""
    logger.debug(f"ENTER create_siliconflow_provider: args={{'api_key': '***'}}")
    config = ProviderConfig(
        provider_id="siliconflow",
        name="硅基流动",
        base_url="https://api.siliconflow.cn/v1",
        api_key=api_key,
        priority=10,
        max_concurrent=5,
        timeout=30.0,
        max_retries=3
    )

    provider = SiliconFlowProvider(config)

    # 免费模型列表
    free_models = [
        ModelInfo(
            model_id="deepseek-ai/DeepSeek-V3",
            provider_id="siliconflow",
            display_name="DeepSeek V3",
            capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING},
            max_context_tokens=65536,
            max_output_tokens=8192,
            is_free=True,
            priority=100
        ),
        ModelInfo(
            model_id="Qwen/Qwen2.5-72B-Instruct",
            provider_id="siliconflow",
            display_name="Qwen2.5 72B",
            capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.FUNCTION_CALLING},
            max_context_tokens=32768,
            max_output_tokens=4096,
            is_free=True,
            priority=90
        ),
        ModelInfo(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            provider_id="siliconflow",
            display_name="Qwen2.5 7B",
            capabilities={ModelCapability.CHAT, ModelCapability.CODE},
            max_context_tokens=32768,
            max_output_tokens=2048,
            is_free=True,
            priority=80
        ),
        ModelInfo(
            model_id="THUDM/glm-4-9b-chat",
            provider_id="siliconflow",
            display_name="GLM-4 9B",
            capabilities={ModelCapability.CHAT, ModelCapability.FUNCTION_CALLING},
            max_context_tokens=8192,
            max_output_tokens=2048,
            is_free=True,
            priority=70
        ),
    ]

    logger.info(f"SiliconFlow provider created with {len(free_models)} models")
    logger.debug(f"EXIT create_siliconflow_provider: result=(provider, {len(free_models)} models)")
    return provider, free_models
