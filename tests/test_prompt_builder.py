"""Tests for the prompt builder service."""

from types import SimpleNamespace

from app.services.prompt_builder import build_system_prompt, build_chat_messages


def _make_agent(**kwargs):
    defaults = {
        "name": "TestBot",
        "personality": {
            "openness": 0.8,
            "extraversion": 0.6,
            "agreeableness": 0.9,
            "conscientiousness": 0.7,
            "neuroticism": 0.3,
        },
        "background": "A friendly test agent",
        "values": ["honesty", "curiosity"],
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_build_system_prompt_basic():
    """System prompt includes system section and instruction."""
    agent = _make_agent()
    prompt = build_system_prompt(agent)
    assert "## System" in prompt
    assert "## Instruction" in prompt
    assert "AI companion" in prompt


def test_build_system_prompt_with_personality():
    """Personality section appears when agent has personality."""
    agent = _make_agent()
    prompt = build_system_prompt(agent)
    assert "## Personality Profile" in prompt
    assert "TestBot" in prompt
    assert "Openness: 0.8" in prompt
    assert "Extraversion: 0.6" in prompt


def test_build_system_prompt_no_personality():
    """No personality section when agent has no personality."""
    agent = _make_agent(personality=None)
    prompt = build_system_prompt(agent)
    assert "## Personality Profile" not in prompt


def test_build_system_prompt_with_emotion():
    """Emotion section appears when emotion dict is provided."""
    agent = _make_agent()
    emotion = {"valence": 0.5, "arousal": 0.3, "dominance": 0.7}
    prompt = build_system_prompt(agent, emotion=emotion)
    assert "## Current Emotion State" in prompt
    assert "Valence: 0.5" in prompt


def test_build_system_prompt_with_memories():
    """Memory section appears with numbered list."""
    agent = _make_agent()
    memories = ["User likes sushi", "User visited Tokyo"]
    prompt = build_system_prompt(agent, memories=memories)
    assert "## Relevant Memories" in prompt
    assert "1. User likes sushi" in prompt
    assert "2. User visited Tokyo" in prompt


def test_build_system_prompt_with_summaries():
    """Summarizer sections appear when summaries provided."""
    agent = _make_agent()
    summaries = {
        "review": "This is a review",
        "distillation": "Key points here",
        "state": "Current state analysis",
    }
    prompt = build_system_prompt(agent, summaries=summaries)
    assert "### Conversation Review" in prompt
    assert "### Memory Distillation" in prompt
    assert "### Current State" in prompt


def test_build_system_prompt_with_graph_context():
    """Graph context section appears with topics and entities."""
    agent = _make_agent()
    graph_context = {
        "topics": ["food", "travel"],
        "entities": ["Tokyo", "sushi"],
    }
    prompt = build_system_prompt(agent, graph_context=graph_context)
    assert "## Relationship Context" in prompt
    assert "food" in prompt
    assert "Tokyo" in prompt


def test_build_chat_messages():
    """Chat messages include system prompt and recent messages."""
    messages = [
        {"role": "user", "content": f"msg {i}"}
        for i in range(10)
    ]
    result = build_chat_messages("System prompt here", messages, max_recent=6)

    assert result[0]["role"] == "system"
    assert result[0]["content"] == "System prompt here"
    assert len(result) == 7  # system + 6 recent
    assert result[-1]["content"] == "msg 9"


def test_build_chat_messages_fewer_than_max():
    """When fewer messages than max, all are included."""
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    result = build_chat_messages("sys", messages)
    assert len(result) == 3  # system + 2
