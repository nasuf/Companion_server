"""Local smoke test for Alibaba Cloud Bailian / DashScope models."""

import asyncio
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


async def main() -> None:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    base_url = os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    chat_model = os.environ.get("CHAT_MODEL") or os.environ.get("REMOTE_CHAT_MODEL") or "qwen3.5-plus"
    small_model = (
        os.environ.get("SUMMARIZER_MODEL")
        or os.environ.get("UTILITY_MODEL")
        or os.environ.get("REMOTE_SMALL_MODEL")
        or "qwen3.5-flash"
    )
    embedding_model = (
        os.environ.get("EMBEDDING_MODEL")
        or os.environ.get("REMOTE_EMBEDDING_MODEL")
        or "text-embedding-v3"
    )
    enable_thinking = os.environ.get("DASHSCOPE_ENABLE_THINKING", "false").lower() == "true"

    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is required")

    big = ChatOpenAI(
        model=chat_model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=256,
        extra_body={"enable_thinking": enable_thinking},
    )
    small = ChatOpenAI(
        model=small_model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=128,
        extra_body={"enable_thinking": enable_thinking},
    )
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        base_url=base_url,
        dimensions=768,
        check_embedding_ctx_length=False,
    )

    big_response = await big.ainvoke("用一句简短中文介绍你自己，不要超过20字。")
    small_response = await small.ainvoke("把“今天心情很差”改写成更自然的口语一句话。")
    vector = await embeddings.aembed_query("我最近和老板吵架，心情很差。")

    print("chat:", big_response.content)
    print("small:", small_response.content)
    print("embedding_dims:", len(vector))


if __name__ == "__main__":
    asyncio.run(main())
