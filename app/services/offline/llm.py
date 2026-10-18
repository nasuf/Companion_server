"""Which models write and review activity copy.

By default ("runtime") they follow the admin model management page: the chat
model writes, the small model reviews. Pinning a provider keeps a separate
route (offline_chat_model / offline_small_model), at the cost of the call
profile: resilience recognises only the runtime chat model as a long-output
caller, so long offline calls pass ``OFFLINE_WRITER_PROFILE`` explicitly.
"""

from functools import lru_cache
from langchain_core.language_models import BaseChatModel

from app.config import settings
from app.services.llm.models import get_chat_model, get_utility_model
from app.services.llm.providers import build_chat_model


# Long-output offline calls (card selection, detail copy, notes). A pinned
# model would otherwise get the 12-second small-model profile: on 2026-10-18
# both card-selection attempts were cut at 12 s and the card fell back.
OFFLINE_WRITER_PROFILE = "chat_extract"


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
