from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from utils.helpers.logger import get_logger

logger = get_logger("llm.providers")

_PROVIDERS = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}

def get_provider(provider_name):
    provider_cls = _PROVIDERS.get(provider_name)
    if provider_cls is None:
        logger.error(f"Unknown provider '{provider_name}', supported: {list(_PROVIDERS.keys())}")
        raise ValueError(f"Unknown provider '{provider_name}', supported: {list(_PROVIDERS.keys())}")
    return provider_cls()
