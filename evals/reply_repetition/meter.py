"""复读门禁的本地计费代理: 回放与打分子进程只拿到这个地址和占位 key, 真 key 只在这里。

每次调用前按请求体字节数 (不少于 token 数) 与 max_tokens 算出最坏费用, 从该模型当前的预留块里扣;
块里放不下就先向服务端预留新块 (每个模型各一块, 服务端按单次与月度上限拒绝)。调用结束按实际
用量记账, 块用完或运行结束时一次结算。结果不明的调用 (超时、5xx、没返回用量) 按最坏费用记,
不当作免费; 供应商在推理前就拒收的 (4xx) 不计费。
"""
from __future__ import annotations

import asyncio
import json
import math
import socket
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from app.services.quality import transport
from app.services.quality.budget import BudgetExhausted

MAX_BODY = 262144  # 主回复提示词实测约 22 KB (UTF-8)
DEFAULT_MAX_TOKENS = 1024  # 打分的复读评判不带 max_tokens; 它只回一行 JSON
MAX_TOKENS = 2048
# 块内累计不超过额度的 99.9%: 服务端按 Decimal 重算, 浮点误差不能顶破预留
SAFETY = 0.999
FREE_STATUS = frozenset({400, 401, 403, 404, 422, 429})
BLOCK_STEPS = (Decimal("2"), Decimal("0.5"), Decimal("0.1"))
JUDGES = ("qwen3.5-flash", "qwen3.5-plus")
EMBEDDING = "text-embedding-v4"


@dataclass
class Block:
    id: str
    amount: float
    committed: float = 0.0
    inflight: int = 0
    draining: bool = False
    settling: bool = False
    input: int = 0
    output: int = 0
    cached_input: int = 0
    calls: int = 0
    unknown_cny: float = 0.0

    def usage(self) -> dict:
        return dict(input=self.input, output=self.output, cached_input=self.cached_input,
                    calls=self.calls, unknown_cny=math.ceil(self.unknown_cny * 1e8) / 1e8)


def _error(status: int, code: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": code}}, status_code=status)


class Meter:
    def __init__(self, control, prices: dict, upstreams: dict, keys: dict, route: list[str], *, client=None):
        self.control = control
        self.prices = prices
        self.upstreams = {k: v.rstrip("/") for k, v in upstreams.items()}
        self.keys = keys
        self.allowed = {(route[0], route[1], "chat/completions"), ("dashscope", EMBEDDING, "embeddings"),
                        *(("dashscope", judge, "chat/completions") for judge in JUDGES)}
        self.blocks: dict[str, Block] = {}
        self.locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.settlements: set[asyncio.Task] = set()
        self.exhausted = False
        self.errors: list[str] = []
        self.stats: dict[str, dict] = defaultdict(lambda: dict(
            calls=0, input=0, output=0, cached_input=0, cost_cny=0.0, unknown_calls=0, unknown_cny=0.0,
            rejected=0, blocks=0))
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(120, connect=15), trust_env=False)
        self.app = Starlette(routes=[Route("/{provider}/{path:path}", self.handle, methods=["POST"])])

    async def _open(self, model: str, bound: float) -> Block:
        for amount in BLOCK_STEPS:
            if float(amount) * SAFETY < bound:
                break
            try:
                ident = await self.control.reserve_block(model, amount)
            except BudgetExhausted:
                continue  # 单次或月度剩余不足一整块时退到小块
            self.stats[model]["blocks"] += 1
            return Block(id=ident, amount=float(amount))
        self.exhausted = True
        raise BudgetExhausted("evaluation_budget_exhausted")

    async def acquire(self, model: str, bound: float) -> Block:
        async with self.locks[model]:
            block = self.blocks.get(model)
            if block is None or block.committed + bound > block.amount * SAFETY:
                if block is not None:
                    del self.blocks[model]
                    block.draining = True
                    self._settle_when_idle(block)
                block = await self._open(model, bound)
                self.blocks[model] = block
            block.committed += bound
            block.inflight += 1
            return block

    def release(self, model: str, block: Block, bound: float, outcome: str, cost: float = 0.0,
                used: dict | None = None) -> None:
        """outcome: billed (按实际用量) / unknown (按最坏费用) / free (供应商拒收)。"""
        stats = self.stats[model]
        block.inflight -= 1
        if outcome == "billed":
            block.committed += cost - bound
            block.calls += 1
            for key in ("input", "output", "cached_input"):
                setattr(block, key, getattr(block, key) + used[key])
                stats[key] += used[key]
            stats["calls"] += 1
            stats["cost_cny"] += cost
        elif outcome == "unknown":
            block.calls += 1
            block.unknown_cny += bound
            stats["unknown_calls"] += 1
            stats["unknown_cny"] += bound
        else:
            block.committed -= bound
            stats["rejected"] += 1
        self._settle_when_idle(block)

    def _settle_when_idle(self, block: Block) -> None:
        if block.draining and not block.inflight and not block.settling:
            block.settling = True
            task = asyncio.create_task(self._settle(block))
            self.settlements.add(task)
            task.add_done_callback(self.settlements.discard)

    async def _settle(self, block: Block) -> None:
        try:
            await self.control.settle_block(block.id, block.usage())
        except Exception as exc:  # 结算不了的块保留整块预留 (服务端按预留额计), 不重放
            self.errors.append("settle_" + type(exc).__name__)
            try:
                await self.control.uncertain(block.id)
            except Exception as again:
                self.errors.append("uncertain_" + type(again).__name__)

    async def close(self) -> None:
        for model, block in list(self.blocks.items()):
            del self.blocks[model]
            block.draining = True
            self._settle_when_idle(block)
        while self.settlements:
            await asyncio.gather(*list(self.settlements))
        await self.client.aclose()

    def summary(self) -> dict:
        models = {m: {**s, "cost_cny": round(s["cost_cny"], 4), "unknown_cny": round(s["unknown_cny"], 4)}
                  for m, s in sorted(self.stats.items())}
        return {"models": models, "errors": self.errors, "exhausted": self.exhausted,
                "cost_cny": round(sum(s["cost_cny"] + s["unknown_cny"] for s in self.stats.values()), 4)}

    async def handle(self, request: Request) -> Response:
        provider, path = request.path_params["provider"], request.path_params["path"]
        raw = await request.body()
        if len(raw) > MAX_BODY:
            return _error(413, "evaluation_request_too_large")
        try:
            body = json.loads(raw)
        except ValueError:
            return _error(400, "evaluation_request_invalid")
        if not isinstance(body, dict) or (provider, body.get("model"), path) not in self.allowed:
            return _error(403, "evaluation_model_not_allowed")
        model = f"{provider}/{body['model']}"
        price = transport.pinned_price(model, self.prices)
        max_tokens = 0
        if path == "chat/completions":
            if body.get("stream"):
                return _error(400, "evaluation_stream_not_metered")
            max_tokens = body.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
            if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_TOKENS:
                return _error(400, "evaluation_max_tokens_out_of_bounds")
        data = json.dumps(body, ensure_ascii=False).encode()
        bound = transport.price_cost(price, len(data), max_tokens)
        # 402: OpenAI SDK 不重试这个状态码 (预算停了, 重试只会空转)
        if self.exhausted:
            return _error(402, "evaluation_budget_exhausted")
        try:
            block = await self.acquire(model, bound)
        except BudgetExhausted:
            return _error(402, "evaluation_budget_exhausted")
        except Exception as exc:
            self.errors.append("reserve_" + type(exc).__name__)
            return _error(503, "evaluation_control_unavailable")
        outcome = "unknown"
        cost, used = 0.0, None
        try:
            upstream = await self.client.post(
                self.upstreams[provider] + "/" + path, content=data,
                headers={"Authorization": "Bearer " + self.keys[provider], "Content-Type": "application/json"})
            if upstream.status_code in FREE_STATUS:
                outcome = "free"
            elif upstream.status_code == 200:
                used = usage(path, upstream, len(data), max_tokens)
                if used is not None:
                    outcome = "billed"
                    cost = transport.price_cost(price, used["input"], used["output"], used["cached_input"])
            return Response(upstream.content, status_code=upstream.status_code, media_type="application/json")
        except httpx.HTTPError:
            return _error(502, "evaluation_upstream_unavailable")
        finally:
            self.release(model, block, bound, outcome, cost, used)


def usage(path: str, response: httpx.Response, input_bound: int, max_tokens: int) -> dict | None:
    """供应商回报的用量; 缺失或超出请求能产生的上限时返回 None (按最坏费用记)。"""
    try:
        value = response.json()["usage"]
        inputs = value["prompt_tokens"] if path == "chat/completions" else value.get("prompt_tokens", value["total_tokens"])
        outputs = value["completion_tokens"] if path == "chat/completions" else 0
        cached = (value.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    except (ValueError, KeyError, TypeError):
        return None
    if any(type(n) is not int for n in (inputs, outputs, cached)):
        return None
    if not (0 <= cached <= inputs <= input_bound and 0 <= outputs <= max_tokens):
        return None
    return dict(input=inputs, output=outputs, cached_input=cached)


async def serve(meter: Meter):
    """在 127.0.0.1 的随机端口上起代理; 返回 (端口, 停止函数)。"""
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(meter.app, log_level="warning", access_log=False, lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        if task.done():
            task.result()
        await asyncio.sleep(0.05)

    async def stop() -> None:
        server.should_exit = True
        await task

    return port, stop
