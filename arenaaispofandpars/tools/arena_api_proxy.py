#!/usr/bin/env python3
"""
Anthropic-совместимый HTTP-прокси над arena-api.

Позволяет использовать библиотеку arena-api в качестве модели в Claude Code,
подставив её вместо Claude:

    ANTHROPIC_BASE_URL=http://localhost:8765 ANTHROPIC_API_KEY=dummy claude

Требования:
    pip install fastapi uvicorn
    # один раз: войти в аккаунт Arena (браузер или cookies)
    arena-api --login
    # или задать ARENA_COOKIE / ARENA_COOKIES_FILE перед запуском прокси

Запуск:
    python tools/arena_api_proxy.py        # http://localhost:8765
    PORT=9000 python tools/arena_api_proxy.py

.. note::
    Это ранний прототип (и ему нужны ``fastapi`` и ``uvicorn``). В самой
    библиотеке теперь есть полноценный шим с тем же протоколом::

        arena-api --claude --port 8082

    Он без внешних зависимостей, ходит через тёплый демон, сохраняет всю
    историю и ``system``, поддерживает ``tool_use`` и ``stop_sequences``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from arena_api.auto import auto_async_direct_client
from arena_api.compat.anthropic_server import resolve_arena_model
from arena_api.exceptions import ModelUnavailableError


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        await get_client()
    except Exception:  # noqa: BLE001 - lazy retry on first request
        pass
    try:
        yield
    finally:
        global _client
        if _client is not None:
            await _client.aclose()
            _client = None


app = FastAPI(title="arena-api anthropic proxy", lifespan=_lifespan)

_PORT_DEFAULT = 8765

_client = None  # singleton AsyncDirectArenaClient (браузер остаётся тёплым)


async def get_client():
    """Создаёт один раз и переиспользует AsyncDirectArenaClient."""
    global _client
    if _client is None:
        _client = await auto_async_direct_client(
            verification_token="auto",
            prewarm=True,
        )
    return _client


def _sse(event: str, data: dict) -> str:
    """Формирует одну SSE-кадру в формате Anthropic Messages API."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extract_prompt(messages: list[dict]) -> str:
    """Берёт текст последнего пользовательского сообщения."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            return "\n".join(parts)
    return ""


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "arena",
                "type": "model",
                "owned_by": "arena-api",
            },
            {
                "id": "claude-3-haiku",
                "type": "model",
                "owned_by": "arena-api",
            },
        ],
    }


@app.post("/v1/messages")
async def messages(request: Request):
    """Anthropic Messages API — точка входа Claude Code."""
    body = await request.json()
    model = body.get("model", "arena")
    arena_model = resolve_arena_model(model)
    stream = body.get("stream", False)
    messages = body.get("messages", [])

    prompt = _extract_prompt(messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="no user message found")

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    if not stream:
        # Непоточный ответ: собираем все чанки и отдаём одним объектом.
        chunks: list[str] = []
        client = await get_client()
        try:
            try:
                async for chunk in client.astream(prompt, model=arena_model):
                    if getattr(chunk, "delta", None):
                        chunks.append(chunk.delta)
            except ModelUnavailableError:
                if arena_model is None:
                    raise
                async for chunk in client.astream(prompt, model=None):
                    if getattr(chunk, "delta", None):
                        chunks.append(chunk.delta)
        except Exception as exc:  # noqa: BLE001 - отдаём понятную ошибку клиенту
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        text = "".join(chunks)
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": max(0, len(text))},
        }

    async def event_generator() -> AsyncIterator[str]:
        client = await get_client()
        try:
            # message_start
            yield _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
            # content_block_start
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            try:
                try:
                    async for chunk in client.astream(prompt, model=arena_model):
                        delta = getattr(chunk, "delta", None) or ""
                        if delta:
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "text_delta", "text": delta},
                                },
                            )
                except ModelUnavailableError:
                    if arena_model is None:
                        raise
                    async for chunk in client.astream(prompt, model=None):
                        delta = getattr(chunk, "delta", None) or ""
                        if delta:
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "text_delta", "text": delta},
                                },
                            )
            finally:
                yield _sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": 0},
                )
                yield _sse("message_stop", {"type": "message_stop"})
        except Exception as exc:  # noqa: BLE001
            # В потоке ошибку сложно отдать клиенту — пишем в лог и закрываем.
            import sys

            print(f"[arena-api proxy] stream error: {exc}", file=sys.stderr)
            yield _sse(
                "error",
                {"type": "error", "error": {"type": "upstream_error", "message": str(exc)}},
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", _PORT_DEFAULT))
    uvicorn.run(app, host="0.0.0.0", port=port)
