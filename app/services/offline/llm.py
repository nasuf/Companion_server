"""Activity extraction/copy models have an explicit, separately tested route."""

from functools import lru_cache
from langchain_core.language_models import BaseChatModel

from app.config import settings
from app.services.llm.models import get_chat_model, get_utility_model
from app.services.llm.providers import build_chat_model


@lru_cache(maxsize=16)
def _model(provider: str, name: str) -> BaseChatModel:
    return build_chat_model(provider, name)


def get_offline_chat_model() -> BaseChatModel:
    if settings.offline_model_provider == "runtime":
        return get_chat_model()
    return _model(settings.offline_model_provider, settings.offline_chat_model)


def get_offline_small_model() -> BaseChatModel:
    if settings.offline_model_provider == "runtime":
        return get_utility_model()
    return _model(settings.offline_model_provider, settings.offline_small_model)
