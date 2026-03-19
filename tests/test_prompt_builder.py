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
        "values": {"gender": "female"},
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_build_system_prompt_basic():
    """System prompt includes core rules and response instruction sections."""
    agent = _make_agent()
    prompt = build_system_prompt(agent)
    assert "## 核心规则" in prompt
    assert "## 回复要求" in prompt


def test_build_system_prompt_with_personality():
    """Personality section appears when agent has personality."""
    agent = _make_agent()
    prompt = build_system_prompt(agent)
    assert "## 你的身份" in prompt
    assert "TestBot" in prompt
    assert "性格" in prompt


def test_build_system_prompt_no_personality():
    """Still has identity section even without personality data."""
    agent = _make_agent(personality=None)
    prompt = build_system_prompt(agent)
    assert "## 你的身份" in prompt
    assert "TestBot" in prompt


def test_build_system_prompt_with_emotion():
    """Emotion section appears when emotion dict is provided."""
    agent = _make_agent()
    emotion = {"pleasure": 0.5, "arousal": 0.3, "dominance": 0.7}
    prompt = build_system_prompt(agent, emotion=emotion)
    assert "## 当前情绪" in prompt
    assert "PAD" in prompt


def test_build_system_prompt_with_memories():
    """Memory section appears with numbered list."""
    agent = _make_agent()
    memories = ["User likes sushi", "User visited Tokyo"]
    prompt = build_system_prompt(agent, memories=memories)
    assert "## 你记得的事情" in prompt
    assert "1. User likes sushi" in prompt
    assert "2. User visited Tokyo" in prompt


def test_build_system_prompt_with_working_facts():
    """Working memory section appears with hot-path facts."""
    agent = _make_agent()
    prompt = build_system_prompt(
        agent,
        working_facts=["[plan] 下周去东京出差", "[preference] 喜欢黑咖啡"],
    )
    assert "## 当前会话事实" in prompt
    assert "下周去东京出差" in prompt


def test_build_system_prompt_with_summaries():
    """Summarizer sections appear when summaries provided."""
    agent = _make_agent()
    summaries = {
        "review": "This is a review",
        "distillation": "Key points here",
        "state": "Current state analysis",
    }
    prompt = build_system_prompt(agent, summaries=summaries)
    assert "### 对话回顾" in prompt
    assert "### 记忆要点" in prompt
    assert "### 当前状态" in prompt


def test_build_system_prompt_with_graph_context():
    """Graph context section appears with topics and entities."""
    agent = _make_agent()
    graph_context = {
        "topics": ["food", "travel"],
        "entities": ["Tokyo", "sushi"],
    }
    prompt = build_system_prompt(agent, graph_context=graph_context)
    assert "## 关系上下文" in prompt
    assert "food" in prompt
    assert "Tokyo" in prompt


def test_build_system_prompt_with_patience():
    """Patience instruction appears when provided."""
    agent = _make_agent()
    prompt = build_system_prompt(agent, patience_instruction="你对用户有些不满")
    assert "## 情绪状态提醒" in prompt
    assert "不满" in prompt


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
