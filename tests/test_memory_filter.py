"""记忆消息过滤器测试。"""

from app.services.memory.recording.filter import should_extract_memory


class TestShouldExtractMemory:
    def test_empty_message(self):
        assert not should_extract_memory("")
        assert not should_extract_memory("   ")

    def test_short_message_no_signal(self):
        """Very short, no keywords → filtered out."""
        assert not should_extract_memory("嗯")
        assert not should_extract_memory("好的")
        assert not should_extract_memory("哈哈")

    def test_self_disclosure(self):
        """First person + fact = should extract."""
        assert should_extract_memory("我是一个程序员，在北京工作")

    def test_emotion_signal(self):
        """Emotion word alone has weight=2, enough to pass."""
        assert should_extract_memory("今天好开心啊真的很幸福")

    def test_time_reference(self):
        """Time + first person = extract."""
        assert should_extract_memory("我昨天去看了电影")

    def test_fact_statement(self):
        """Fact + first person = extract."""
        assert should_extract_memory("我在上海住了三年了")

    def test_short_core_profile(self):
        """Short but high-value profile facts should pass."""
        assert should_extract_memory("我28岁")
        assert should_extract_memory("我叫小明")

    def test_english_self_disclosure(self):
        """English self-disclosure should not be filtered out."""
        assert should_extract_memory("I work in San Francisco")
        assert should_extract_memory("My birthday is in July")

    def test_long_message(self):
        """Long message has inherent value."""
        msg = "今天天气真的非常好，阳光明媚，我和朋友们一起去公园散步，心情特别愉快"
        assert should_extract_memory(msg)

    def test_pure_greeting(self):
        """Greetings should be filtered."""
        assert not should_extract_memory("你好")
        assert not should_extract_memory("早上好")

    def test_question_only(self):
        """Simple questions without self-info."""
        assert not should_extract_memory("你呢")
        assert not should_extract_memory("是吗")
