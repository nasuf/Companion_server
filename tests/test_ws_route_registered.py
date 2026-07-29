"""WebSocket 路由必须真的挂在端点函数上.

2026-07-29 生产事故: 往 ws.py 里新增一个辅助函数时, 它被插到了
`@router.websocket("/ws/{conversation_id}")` 与 `websocket_endpoint` 之间 ——
装饰器于是装到了那个辅助函数上, 真正的端点完全没注册。

表现是全端聊天不可用 (web 与 flutter 都停在"重连中"), 而**日志里没有任何异常**:
Starlette 对未匹配的 WebSocket 路由就是静默拒绝, uvicorn 记一行
"connection rejected (403 Forbidden)"。健康检查照常通过, HTTP 接口全部正常。

整套测试 3400+ 条, 没有一条验证过"这个路由存在" —— 因为它一直存在, 没人想过它会
消失。这个文件就是补这个缺口。
"""

from __future__ import annotations

import ast
from pathlib import Path

_WS = Path(__file__).resolve().parents[1] / "app" / "api" / "realtime" / "ws.py"


def _decorated_websocket_handlers() -> dict[str, str]:
    """返回 {路径: 被装饰的函数名}."""
    tree = ast.parse(_WS.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            if not isinstance(deco, ast.Call):
                continue
            fn = deco.func
            if getattr(fn, "attr", None) != "websocket":
                continue
            if deco.args and isinstance(deco.args[0], ast.Constant):
                out[deco.args[0].value] = node.name
    return out


def test_chat_websocket_route_exists():
    handlers = _decorated_websocket_handlers()
    assert "/ws/{conversation_id}" in handlers, (
        "聊天 WebSocket 路由没有注册 —— 全端聊天会停在「重连中」, 而日志里只有一行 "
        "403, 没有任何异常"
    )


def test_route_is_bound_to_the_real_endpoint():
    """装饰器必须装在真正的端点上, 而不是碰巧排在它前面的某个函数.

    往 ws.py 里插新函数时极容易插进装饰器和端点之间 —— 语法完全合法, 测试全绿,
    只有真正连一次 WebSocket 才会发现。
    """
    handlers = _decorated_websocket_handlers()
    handler = handlers.get("/ws/{conversation_id}")
    assert handler == "websocket_endpoint", (
        f"/ws/{{conversation_id}} 现在指向 {handler!r}, 而不是 websocket_endpoint。"
        "多半是新函数被插到了装饰器与端点之间。"
    )


def test_endpoint_signature_matches_the_route():
    """端点必须收 websocket 和路径参数, 否则 Starlette 调用时会直接失败."""
    tree = ast.parse(_WS.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "websocket_endpoint"
    )
    args = [a.arg for a in fn.args.args]
    assert args[:2] == ["websocket", "conversation_id"], (
        f"websocket_endpoint 的参数是 {args}, 与路由 /ws/{{conversation_id}} 对不上"
    )


def test_helper_functions_are_not_accidentally_decorated():
    """下划线开头的内部函数不该挂着任何 websocket 路由."""
    bad = {
        path: name for path, name in _decorated_websocket_handlers().items()
        if name.startswith("_")
    }
    assert not bad, f"内部辅助函数被误挂成了 WebSocket 端点: {bad}"
