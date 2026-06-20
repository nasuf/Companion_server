from app.services.chat_links.cards import component_card_for_link, metadata_for_link_card
from app.services.chat_links.extraction import (
    LinkMetadata,
    extract_first_url,
    extract_link_metadata,
    extract_urls,
    platform_for_url,
)
from app.services.chat_links.repo import (
    ChatLinkCard,
    bind_link_card_to_message,
    create_or_update_link_card,
    find_link_card,
    list_user_link_groups,
)
from app.services.chat_links.recommendation import (
    ProactiveLinkRecommendation,
    configured_candidate_urls,
    maybe_prepare_proactive_link_recommendation,
    should_attempt_proactive_link,
)

__all__ = [
    "ChatLinkCard",
    "LinkMetadata",
    "ProactiveLinkRecommendation",
    "bind_link_card_to_message",
    "component_card_for_link",
    "configured_candidate_urls",
    "create_or_update_link_card",
    "extract_first_url",
    "extract_link_metadata",
    "extract_urls",
    "find_link_card",
    "list_user_link_groups",
    "maybe_prepare_proactive_link_recommendation",
    "metadata_for_link_card",
    "platform_for_url",
    "should_attempt_proactive_link",
]
