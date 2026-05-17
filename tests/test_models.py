"""
Hermes Cluster v6.0 - 多模型路由测试
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hermes_cluster.models import (
    ModelCapability, ProviderStatus, RouteStrategy,
    ModelInfo, ProviderConfig, ChatMessage, ChatRequest, ChatResponse,
    UsageRecord, SiliconFlowProvider, OpenAICompatibleProvider,
    ModelRouter, create_siliconflow_provider,
)


class TestModelInfo:
    """模型信息测试"""

    def test_model_info_creation(self):
        model = ModelInfo(
            model_id="test-model",
            provider_id="test-provider",
            display_name="Test Model",
            capabilities={ModelCapability.CHAT, ModelCapability.CODE},
            max_context_tokens=8192,
            is_free=True,
            priority=10
        )
        assert model.model_id == "test-model"
        assert ModelCapability.CHAT in model.capabilities
        assert model.is_free == True

    def test_model_info_defaults(self):
        model = ModelInfo(
            model_id="minimal",
            provider_id="p1",
            display_name="Minimal"
        )
        assert model.capabilities == set()
        assert model.max_context_tokens == 4096
        assert model.cost_per_1k_input == 0.0


class TestChatRequest:
    """聊天请求测试"""

    def test_chat_request_creation(self):
        req = ChatRequest(
            model="test-model",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello"),
            ],
            temperature=0.5,
            max_tokens=1024
        )
        assert len(req.messages) == 2
        assert req.stream == False
        assert req.request_id is not None

    def test_chat_request_with_tools(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }]
        req = ChatRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Weather?")],
            tools=tools
        )
        assert len(req.tools) == 1


class TestModelRouter:
    """模型路由器测试"""

    def _create_mock_provider(self, provider_id: str = "mock", status: ProviderStatus = ProviderStatus.ACTIVE):
        config = ProviderConfig(
            provider_id=provider_id,
            name=f"Mock {provider_id}",
            base_url="https://mock.api/v1",
            api_key="mock-key",
            status=status,
            priority=10
        )
        provider = OpenAICompatibleProvider(config)
        return provider

    def test_register_provider(self):
        router = ModelRouter()
        provider = self._create_mock_provider()
        router.register_provider(provider)
        assert "mock" in router._providers

    def test_register_model(self):
        router = ModelRouter()
        model = ModelInfo(
            model_id="test-model",
            provider_id="mock",
            display_name="Test"
        )
        router.register_model(model)
        assert "test-model" in router._models
        assert router._model_to_provider["test-model"] == "mock"

    @pytest.mark.asyncio
    async def test_route_by_priority(self):
        router = ModelRouter()
        router.set_strategy(RouteStrategy.PRIORITY)

        p1 = self._create_mock_provider("p1")
        p2 = self._create_mock_provider("p2")
        router.register_provider(p1)
        router.register_provider(p2)

        router.register_model(ModelInfo("low-prio", "p1", "Low", priority=1))
        router.register_model(ModelInfo("high-prio", "p2", "High", priority=100))

        selected = await router.route()
        assert selected.model_id == "high-prio"

    @pytest.mark.asyncio
    async def test_route_by_least_cost(self):
        router = ModelRouter()
        router.set_strategy(RouteStrategy.LEAST_COST)

        p = self._create_mock_provider()
        router.register_provider(p)

        router.register_model(ModelInfo("expensive", "mock", "Expensive", cost_per_1k_input=10.0))
        router.register_model(ModelInfo("cheap", "mock", "Cheap", cost_per_1k_input=1.0))

        selected = await router.route()
        assert selected.model_id == "cheap"

    @pytest.mark.asyncio
    async def test_route_by_capability(self):
        router = ModelRouter()

        p = self._create_mock_provider()
        router.register_provider(p)

        router.register_model(ModelInfo("chat-only", "mock", "ChatOnly", capabilities={ModelCapability.CHAT}))
        router.register_model(ModelInfo("code-model", "mock", "CodeModel", capabilities={ModelCapability.CHAT, ModelCapability.CODE}))

        selected = await router.route(capabilities={ModelCapability.CODE})
        assert selected.model_id == "code-model"

    @pytest.mark.asyncio
    async def test_route_specific_model(self):
        router = ModelRouter()
        p = self._create_mock_provider()
        router.register_provider(p)
        router.register_model(ModelInfo("target-model", "mock", "Target"))

        selected = await router.route(model="target-model")
        assert selected.model_id == "target-model"

    @pytest.mark.asyncio
    async def test_route_no_available(self):
        router = ModelRouter()
        selected = await router.route()
        assert selected is None

    @pytest.mark.asyncio
    async def test_route_skip_disabled_provider(self):
        router = ModelRouter()

        p_active = self._create_mock_provider("active", ProviderStatus.ACTIVE)
        p_disabled = self._create_mock_provider("disabled", ProviderStatus.DISABLED)
        router.register_provider(p_active)
        router.register_provider(p_disabled)

        router.register_model(ModelInfo("active-model", "active", "Active"))
        router.register_model(ModelInfo("disabled-model", "disabled", "Disabled"))

        selected = await router.route()
        assert selected.model_id == "active-model"

    @pytest.mark.asyncio
    async def test_round_robin_routing(self):
        router = ModelRouter()
        router.set_strategy(RouteStrategy.ROUND_ROBIN)

        p = self._create_mock_provider()
        router.register_provider(p)

        for i in range(3):
            router.register_model(ModelInfo(f"model-{i}", "mock", f"Model {i}"))

        results = []
        for _ in range(6):
            selected = await router.route()
            results.append(selected.model_id)

        # 应该轮询
        assert results[0] != results[1]
        assert results[1] != results[2]

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        router = ModelRouter()

        p1 = self._create_mock_provider("p1")
        p2 = self._create_mock_provider("p2")
        router.register_provider(p1)
        router.register_provider(p2)

        router.register_model(ModelInfo("primary", "p1", "Primary", priority=100))
        router.register_model(ModelInfo("backup", "p2", "Backup", priority=1))

        fallback = await router._fallback_route("primary")
        assert fallback.model_id == "backup"

    def test_usage_stats(self):
        router = ModelRouter()
        router._usage_history = [
            UsageRecord("p1", "m1", 100, 50, 150, 0.01),
            UsageRecord("p1", "m1", 200, 100, 300, 0.02),
            UsageRecord("p2", "m2", 50, 25, 75, 0.005),
        ]

        stats = router.get_usage_stats()
        assert stats["total_requests"] == 3
        assert stats["total_tokens"] == 525
        assert "p1" in stats["by_provider"]
        assert stats["by_provider"]["p1"]["tokens"] == 450

    def test_latency_stats(self):
        router = ModelRouter()
        router._latency_history = {
            "m1": [10.0, 20.0, 30.0],
            "m2": [5.0, 15.0]
        }

        stats = router.get_latency_stats()
        assert stats["m1"]["avg_ms"] == 20.0
        assert stats["m1"]["min_ms"] == 10.0
        assert stats["m1"]["max_ms"] == 30.0
        assert stats["m1"]["samples"] == 3


class TestCreateSiliconFlowProvider:
    """创建硅基流动提供商测试"""

    def test_create_provider(self):
        provider, models = create_siliconflow_provider("test-key")
        assert provider.provider_id == "siliconflow"
        assert len(models) >= 3

    def test_free_models_list(self):
        _, models = create_siliconflow_provider("test-key")
        model_ids = [m.model_id for m in models]

        assert "deepseek-ai/DeepSeek-V3" in model_ids
        assert "Qwen/Qwen2.5-72B-Instruct" in model_ids

    def test_free_models_are_free(self):
        _, models = create_siliconflow_provider("test-key")
        for m in models:
            assert m.is_free == True

    def test_models_have_capabilities(self):
        _, models = create_siliconflow_provider("test-key")
        deepseek = next(m for m in models if "DeepSeek-V3" in m.model_id)
        assert ModelCapability.CHAT in deepseek.capabilities
        assert ModelCapability.CODE in deepseek.capabilities


class TestChatResponse:
    """聊天响应测试"""

    def test_response_creation(self):
        resp = ChatResponse(
            request_id="req-1",
            model="test-model",
            provider_id="test-provider",
            content="Hello!",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            latency_ms=100.5
        )
        assert resp.content == "Hello!"
        assert resp.usage["total_tokens"] == 15

    def test_response_with_tool_calls(self):
        tool_calls = [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'}
        }]
        resp = ChatResponse(
            request_id="req-2",
            model="test",
            provider_id="test",
            tool_calls=tool_calls
        )
        assert len(resp.tool_calls) == 1


class TestProviderConfig:
    """提供商配置测试"""

    def test_config_defaults(self):
        config = ProviderConfig(
            provider_id="test",
            name="Test",
            base_url="https://test.api/v1",
            api_key="key"
        )
        assert config.status == ProviderStatus.ACTIVE
        assert config.max_concurrent == 10
        assert config.timeout == 30.0
        assert config.max_retries == 3

    def test_config_custom(self):
        config = ProviderConfig(
            provider_id="custom",
            name="Custom",
            base_url="https://custom.api/v1",
            api_key="key",
            max_concurrent=20,
            timeout=60.0,
            max_retries=5,
            priority=100
        )
        assert config.max_concurrent == 20
        assert config.timeout == 60.0
        assert config.priority == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
