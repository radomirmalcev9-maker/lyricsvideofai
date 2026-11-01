from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

import arena_api
from arena_api import (
    AsyncDirectArenaClient,
    DirectArenaClient,
    ModelInfo,
    model_chat_score,
    model_fast_score,
)
from arena_api import _direct_page as direct_page
from arena_api import auto as auto_module
from arena_api import warm
from arena_api.compat import anthropic_server as claude_server
from arena_api.direct import _models_from_markup


def _cap_meta(
    name: str,
    chat_rank: int | None,
    *,
    selectable: bool = True,
    text_in: bool = True,
    text_out: bool = True,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "publicName": name,
        "userSelectable": selectable,
        "capabilities": {
            "inputCapabilities": {"text": text_in},
            "outputCapabilities": {"text": text_out},
        },
    }
    if chat_rank is not None:
        meta["rankByModality"] = {"chat": chat_rank}
    return meta


def test_model_fast_score_ranks_low_latency_text_models_ahead_of_heavy_thinking() -> None:
    opus_thinking = ModelInfo(
        id="11111111-1111-1111-1111-111111111111",
        display_name="claude-opus-4-6-thinking",
        metadata=_cap_meta("claude-opus-4-6-thinking", 1),
    )
    flash_thinking = ModelInfo(
        id="22222222-2222-2222-2222-222222222222",
        display_name="gemini-2.5-flash-thinking",
        metadata=_cap_meta("gemini-2.5-flash-thinking", 4),
    )
    sonnet = ModelInfo(
        id="33333333-3333-3333-3333-333333333333",
        display_name="claude-sonnet-4-6",
        metadata=_cap_meta("claude-sonnet-4-6", 3),
    )
    haiku = ModelInfo(
        id="44444444-4444-4444-4444-444444444444",
        display_name="claude-haiku-4-5-20251001",
        metadata=_cap_meta("claude-haiku-4-5-20251001", 14),
    )
    flash_lite = ModelInfo(
        id="55555555-5555-5555-5555-555555555555",
        display_name="gemini-2.5-flash-lite",
        metadata=_cap_meta("gemini-2.5-flash-lite", 18),
    )
    hidden_fast = ModelInfo(
        id="66666666-6666-6666-6666-666666666666",
        display_name="gemini-2.5-flash-lite-hidden",
        metadata=_cap_meta("gemini-2.5-flash-lite-hidden", 1, selectable=False),
    )
    image_fast = ModelInfo(
        id="77777777-7777-7777-7777-777777777777",
        display_name="flux-1-schnell-fast",
        metadata=_cap_meta("flux-1-schnell-fast", 1, text_out=False),
    )

    catalog = [
        opus_thinking,
        flash_thinking,
        sonnet,
        haiku,
        flash_lite,
        hidden_fast,
        image_fast,
    ]

    # Top-ELO score picks rank #1 (opus-thinking), whereas fast score picks
    # selectable text flash-lite -> haiku -> sonnet -> thinking, and still
    # rejects hidden/image models.
    assert min(catalog, key=model_chat_score).id == opus_thinking.id
    ranked_fast = [m.id for m in sorted(catalog, key=model_fast_score)]
    assert ranked_fast[:5] == [
        flash_lite.id,
        haiku.id,
        sonnet.id,
        opus_thinking.id,
        flash_thinking.id,
    ]
    assert model_fast_score(image_fast) >= 50_000
    assert model_fast_score(hidden_fast) >= 1_000_000
    assert arena_api.model_fast_score is model_fast_score
    assert arena_api.model_chat_score is model_chat_score


def test_direct_clients_use_fast_model_by_default_and_support_max_opt_out() -> None:
    slow_id = "11111111-1111-1111-1111-111111111111"
    fast_id = "22222222-2222-2222-2222-222222222222"
    page_html = (
        '<html><script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(
            {
                "props": {
                    "pageProps": {
                        "initialModels": [
                            {
                                "id": slow_id,
                                **_cap_meta("claude-opus-4-6-thinking", 1),
                            },
                            {
                                "id": fast_id,
                                **_cap_meta("gemini-2.5-flash", 6),
                            },
                        ]
                    }
                }
            }
        )
        + "</script></html>"
    )
    chosen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/text/direct":
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/html"},
                content=page_html.encode("utf-8"),
            )
        if request.url.path == "/nextjs-api/stream/create-evaluation":
            payload = json.loads(request.content)
            chosen_models.append(payload["modelAId"])
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/plain"},
                content=b'a0:"ok"\nad:{"finishReason":"stop"}\n',
            )
        raise AssertionError(request.url)

    with DirectArenaClient(
        base_url="https://arena.test",
        verification_token="tok",
        http_transport=httpx.MockTransport(handler),
    ) as client:
        assert client.complete("hi") == "ok"
        assert client.complete("hi", model="max") == "ok"
        assert client.complete("hi", model="arena-fast") == "ok"

    with DirectArenaClient(
        base_url="https://arena.test",
        verification_token="tok",
        http_transport=httpx.MockTransport(handler),
        fast_model=False,
    ) as max_client:
        assert max_client.complete("hi") == "ok"

    async def _run_async() -> None:
        class _AsyncMock(httpx.AsyncBaseTransport):
            async def handle_async_request(
                self, request: httpx.Request
            ) -> httpx.Response:
                return handler(request)

        async with AsyncDirectArenaClient(
            base_url="https://arena.test",
            verification_token="tok",
            http_transport=_AsyncMock(),
        ) as aclient:
            assert await aclient.acomplete("hi") == "ok"
            assert await aclient.acomplete("hi", model="arena-max") == "ok"

    asyncio.run(_run_async())
    assert chosen_models == [
        fast_id,
        slow_id,
        fast_id,
        slow_id,
        fast_id,
        slow_id,
    ]


def test_models_from_markup_skips_minified_js_braces_in_milliseconds() -> None:
    js_bundle = "function a(b){if(b){for(let i=0;i<10;i++){const x={k:i};}}} " * 4500
    flight = json.dumps(
        '12:["$","$L1",null,{"initialModels":['
        '{"id":"22222222-2222-2222-2222-222222222222","publicName":"gemini-2.5-flash",'
        '"userSelectable":true,"capabilities":{"inputCapabilities":{"text":true},'
        '"outputCapabilities":{"text":true}},"rankByModality":{"chat":3}},'
        '{"id":"11111111-1111-1111-1111-111111111111","publicName":"claude-opus-4-6-thinking",'
        '"userSelectable":true,"capabilities":{"inputCapabilities":{"text":true},'
        '"outputCapabilities":{"text":true}},"rankByModality":{"chat":1}}'
        "]}]"
    )
    markup = (
        f"<!doctype html><html><head><script>{js_bundle}</script>"
        f"<script>self.__next_f.push([1,{flight}])</script></head>"
        f"<body>{'<div>{prose}</div>' * 2000}</body></html>"
    )
    started = time.perf_counter()
    models = _models_from_markup(markup)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert len(models) == 2
    assert {m.display_name for m in models} == {
        "gemini-2.5-flash",
        "claude-opus-4-6-thinking",
    }
    # Previously took ~4,600 ms on this exact shape; now runs in ~10 ms.
    assert elapsed_ms < 250.0, f"_models_from_markup regressed: {elapsed_ms:.1f} ms"


def test_v8_composer_js_extracts_fast_model_and_body_patch_updates_dynamically() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node binary not installed")

    patch_turn_1 = direct_page._build_main_frame_body_patch_js(
        {"modelAId": "model-1", "generationConfig": {"temperature": 0.1}}
    )
    patch_turn_2 = direct_page._build_main_frame_body_patch_js(
        {"modelAId": "model-2"}
    )
    clear_js = direct_page._CLEAR_MAIN_FRAME_BODY_PATCH_JS
    find_composer_js = direct_page._FIND_COMPOSER_JS

    flight_chunk = json.dumps(
        '9:["$","$L2",null,{"initialModels":['
        '{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","publicName":"claude-opus-4-6-thinking",'
        '"userSelectable":true,"capabilities":{"inputCapabilities":{"text":true},"outputCapabilities":{"text":true}},'
        '"rankByModality":{"chat":1}},'
        '{"id":"bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb","publicName":"veo-3-fast",'
        '"userSelectable":true,"capabilities":{"inputCapabilities":{"text":true},"outputCapabilities":{"video":true}},'
        '"rankByModality":{"chat":2}},'
        '{"id":"cccccccc-cccc-cccc-cccc-cccccccccccc","publicName":"gemini-2.5-flash-hidden",'
        '"userSelectable":false,"capabilities":{"inputCapabilities":{"text":true},"outputCapabilities":{"text":true}},'
        '"rankByModality":{"chat":1}},'
        '{"id":"dddddddd-dddd-dddd-dddd-dddddddddddd","publicName":"gemini-2.5-flash",'
        '"userSelectable":true,"capabilities":{"inputCapabilities":{"text":true},"outputCapabilities":{"text":true}},'
        '"rankByModality":{"chat":5}}'
        "]}]"
    )

    script_body_literal = json.dumps(f"self.__next_f.push([1,{flight_chunk}])")

    harness = f"""
    (async () => {{
      const sentBodies = [];
      global.window = {{
        fetch: async (url, init) => {{
          sentBodies.push(JSON.parse(init.body));
          return {{ ok: true }};
        }},
        getComputedStyle: () => ({{ display: "block", visibility: "visible", opacity: "1" }})
      }};
      const textarea = {{
        id: "prompt",
        tagName: "TEXTAREA",
        disabled: false,
        readOnly: false,
        attrs: {{}},
        getAttribute(k) {{ return this.attrs[k] || null; }},
        setAttribute(k, v) {{ this.attrs[k] = v; }},
        removeAttribute(k) {{ delete this.attrs[k]; }},
        closest(sel) {{ return sel === "form" ? {{}} : null; }},
        getBoundingClientRect() {{ return {{ left: 100, top: 200, width: 600, height: 80 }}; }}
      }};
      global.document = {{
        scripts: [
          {{ type: "", textContent: {script_body_literal} }}
        ],
        querySelectorAll(sel) {{
          if (sel.indexOf("textarea") !== -1) return [textarea];
          return [];
        }}
      }};

      const composerInfo = await ({find_composer_js})();

      // Turn 1 patch
      ({patch_turn_1})();
      await window.fetch("/nextjs-api/stream/create-evaluation", {{
        method: "POST",
        body: JSON.stringify({{ modelAId: "orig", userMessage: {{ content: "1" }} }})
      }});

      // Turn 2 patch on the SAME warm page must override Turn 1 (no stale closure!)
      ({patch_turn_2})();
      await window.fetch("/nextjs-api/stream/create-evaluation", {{
        method: "POST",
        body: JSON.stringify({{ modelAId: "orig", userMessage: {{ content: "2" }} }})
      }});

      // Turn 3 cleared patch leaves native body untouched
      ({clear_js})();
      await window.fetch("/nextjs-api/stream/create-evaluation", {{
        method: "POST",
        body: JSON.stringify({{ modelAId: "orig", userMessage: {{ content: "3" }} }})
      }});

      console.log(JSON.stringify({{ composerInfo, sentBodies }}));
    }})();
    """
    proc = subprocess.run(
        [node, "-e", harness],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    out = json.loads(proc.stdout.strip())
    assert out["composerInfo"]["found"] is True
    assert (
        out["composerInfo"]["fastModelId"]
        == "dddddddd-dddd-dddd-dddd-dddddddddddd"
    )
    assert out["sentBodies"] == [
        {
            "modelAId": "model-1",
            "generationConfig": {"temperature": 0.1},
            "userMessage": {"content": "1"},
        },
        {
            "modelAId": "model-2",
            "userMessage": {"content": "2"},
        },
        {
            "modelAId": "orig",
            "userMessage": {"content": "3"},
        },
    ]


def test_auto_fast_model_self_heals_on_404_and_warm_archive_is_non_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast_uuid = "dddddddd-dddd-dddd-dddd-dddddddddddd"
    archive_started = asyncio.Event()
    archive_aborted = False
    patched_models: list[str | None] = []
    attempt_counter = 0

    class FakeResponse:
        def __init__(self, url: str, status: int, body: str) -> None:
            self.url = url
            self.status = status
            self.headers = {"content-type": "text/plain; charset=utf-8"}
            self.request = type("R", (), {"method": "POST"})()
            self._body = body

        async def text(self) -> str:
            return self._body

    class FakePage:
        def __init__(self, *, fail_fast_model_once: bool) -> None:
            self._fail_fast_once = fail_fast_model_once
            self._listeners: dict[str, list[Any]] = {}
            self._active_patch: dict[str, Any] | None = None
            self._fast_id: str | None = fast_uuid
            self.keyboard = type(
                "K", (), {"type": self._type}
            )()
            self.mouse = type("M", (), {"move": self._noop})()
            self.context = type(
                "C", (), {"add_cookies": self._noop, "cookies": self._cookies}
            )()
            self.url = "https://arena.ai/text/direct?model_a=max"
            self.typed_delays: list[int] = []

        def is_closed(self) -> bool:
            return False

        async def _type(self, text: str, delay: int = 0) -> None:
            del text
            self.typed_delays.append(delay)

        async def _noop(self, *a: Any, **kw: Any) -> None:
            return None

        async def _cookies(self, urls: Any = None) -> list[dict[str, str]]:
            del urls
            return [{"name": "arena-auth-prod-v1.0", "value": "fresh-jwt"}]

        def on(self, event: str, cb: Any) -> None:
            self._listeners.setdefault(event, []).append(cb)

        def remove_listener(self, event: str, cb: Any) -> None:
            cbs = self._listeners.get(event, [])
            if cb in cbs:
                cbs.remove(cb)

        def locator(self, selector: str) -> Any:
            page = self

            class Loc:
                @property
                def first(self) -> "Loc":
                    return self

                async def focus(self) -> None:
                    return None

                async def fill(self, value: str) -> None:
                    del value

                async def click(self, **kw: Any) -> None:
                    del kw
                    if "data-arena-send" in selector:
                        page._emit_eval()

            return Loc()

        async def goto(self, url: str, **kw: Any) -> None:
            del kw
            self.url = url

        async def title(self) -> str:
            return "Arena"

        async def wait_for_timeout(self, ms: int) -> None:
            await asyncio.sleep(ms / 1000.0)

        async def evaluate(self, script: str) -> Any:
            if "window.__arenaFastModelId = null" in script:
                self._fast_id = None
                self._active_patch = None
                return None
            if "window.__arenaBodyPatch =" in script:
                marker = "window.__arenaBodyPatch ="
                line = script.split(marker, 1)[1].split(";", 1)[0].strip()
                self._active_patch = None if line == "null" else json.loads(line)
                return None
            if "data-arena-composer" in script and "data-arena-send" not in script:
                return {
                    "found": True,
                    "tag": "textarea",
                    "x": 400,
                    "y": 700,
                    "fastModelId": self._fast_id,
                }
            if "composer_nonempty" in script:
                return True
            if "data-arena-send" in script:
                return {"found": True, "confidence": "explicit", "x": 650, "y": 700}
            return None

        def _emit_eval(self) -> None:
            nonlocal attempt_counter
            attempt_counter += 1
            model_used = (
                self._active_patch.get("modelAId")
                if isinstance(self._active_patch, dict)
                else None
            )
            patched_models.append(model_used)
            url = "https://arena.ai/nextjs-api/stream/create-evaluation"
            req = type("Req", (), {"method": "POST", "url": url, "post_data": "{}"})()
            for cb in list(self._listeners.get("request", ())):
                cb(req)
            if self._fail_fast_once and model_used == fast_uuid:
                self._fail_fast_once = False
                resp = FakeResponse(url, 404, '{"error":"model not found"}')
            else:
                resp = FakeResponse(
                    url, 200, 'a0:"Быстро!"\nad:{"finishReason":"stop"}\n'
                )
            for cb in list(self._listeners.get("response", ())):
                cb(resp)

    fake_page = FakePage(fail_fast_model_once=True)

    class FakeEnv:
        def __init__(self, options: Any) -> None:
            del options
            self.challenge_handler = type("H", (), {"solve_on_detection": True})()

        async def __aenter__(self) -> "FakeEnv":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def new_page(self) -> FakePage:
            return fake_page

    async def slow_archive(page: Any, *, should_abort: Any = None, **kw: Any) -> bool:
        del page, kw
        nonlocal archive_aborted
        archive_started.set()
        for _ in range(40):
            if callable(should_abort) and should_abort():
                archive_aborted = True
                return False
            await asyncio.sleep(0.015)
        return True

    monkeypatch.setattr(auto_module, "BrowserEnvironment", FakeEnv)
    monkeypatch.setattr(auto_module, "solver_dependencies_available", lambda _: True)
    monkeypatch.setattr(auto_module, "archive_conversation", slow_archive)

    opts = auto_module.AutoOptions(
        base_url="https://arena.ai",
        interactive=False,
        archive_after_answer=True,
        storage_path=tmp_path / "cookies.json",
        use_browser_profiles=False,
    )
    tok = auto_module.DirectVerificationToken(
        opts,
        session_cookies={"arena-auth-prod-v1.0": "jwt"},
        keep_browser_alive=True,
    )
    with DirectArenaClient(
        base_url="https://arena.ai",
        cookies={"arena-auth-prod-v1.0": "jwt"},
        verification_token=tok,
    ) as client:
        t0 = time.perf_counter()
        # Turn 1 tries fast_uuid -> gets 404 -> self-heals to native None -> returns
        # immediately without blocking on slow_archive (600 ms)!
        assert client.complete("Привет") == "Быстро!"
        turn1_ms = (time.perf_counter() - t0) * 1000.0
        assert turn1_ms < 250.0, f"Turn 1 blocked on archive: {turn1_ms:.1f} ms"
        assert patched_models == [fast_uuid, None]
        assert all(d == 0 for d in fake_page.typed_delays)

        # Turn 2 arrives immediately while slow_archive is still running:
        # slow_archive is aborted cleanly instead of stalling Turn 2!
        t1 = time.perf_counter()
        assert client.complete("Второй запрос") == "Быстро!"
        turn2_ms = (time.perf_counter() - t1) * 1000.0
        assert turn2_ms < 250.0, f"Turn 2 waited on archive: {turn2_ms:.1f} ms"
        assert archive_aborted is True
        assert patched_models == [fast_uuid, None, None]


def test_warm_connection_reuses_single_tcp_nodelay_socket_and_forwards_config(
    tmp_path: Path,
) -> None:
    accepted_handshakes = 0
    seen_configs: list[dict[str, Any] | None] = []
    nodelay_flags: list[int] = []

    class FakeWarmSession:
        _accepts_config = True

        async def ask(
            self,
            prompt: str,
            model: str | None = None,
            *,
            config: dict[str, Any] | None = None,
        ):
            seen_configs.append(config)
            yield f"{prompt}:{model or 'default'}"

        async def aclose(self) -> None:
            return None

    state_file = tmp_path / "warm-daemon.json"

    async def _run() -> None:
        server = warm._WarmServer(
            FakeWarmSession(),
            state_file=state_file,
            token="secret-token",
        )
        orig_dispatch = server._dispatch

        async def counting_dispatch(reader: Any, writer: Any) -> None:
            nonlocal accepted_handshakes
            accepted_handshakes += 1
            raw_sock = writer.get_extra_info("socket")
            if raw_sock is not None:
                nodelay_flags.append(
                    int(raw_sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY))
                )
            await orig_dispatch(reader, writer)

        server._dispatch = counting_dispatch  # type: ignore[method-assign]
        state = await server.start()
        try:
            conn = warm.WarmConnection()
            try:
                calls = 0

                def ensure_state(*, force_refresh: bool = False) -> warm.DaemonState:
                    nonlocal calls
                    del force_refresh
                    calls += 1
                    return state

                def _exercise_sync() -> list[str]:
                    results: list[str] = []
                    for idx in range(4):
                        pieces = list(
                            conn.ask_stream(
                                f"q{idx}",
                                ensure_state=ensure_state,
                                model="fast" if idx % 2 == 0 else None,
                                config={"max_tokens": 128} if idx == 3 else None,
                            )
                        )
                        results.append("".join(pieces))
                    assert conn._sock is not None
                    client_nodelay = int(
                        conn._sock.getsockopt(
                            socket.IPPROTO_TCP, socket.TCP_NODELAY
                        )
                    )
                    assert client_nodelay != 0
                    return results

                outputs = await asyncio.to_thread(_exercise_sync)
                assert outputs == [
                    "q0:fast",
                    "q1:default",
                    "q2:fast",
                    "q3:default",
                ]
                # All 4 prompts reused ONE TCP connection and ONE ensure_state call!
                assert accepted_handshakes == 1
                assert calls == 1
                assert seen_configs == [None, None, None, {"max_tokens": 128}]
                assert nodelay_flags and nodelay_flags[0] != 0
            finally:
                conn.close()
        finally:
            await server._shutdown()

    asyncio.run(_run())


def test_claude_shim_streams_immediately_unblocks_code_fences_and_caches_unknown_models() -> None:
    # 1. stream_window unblocks non-JSON code fences while preserving <tool & ```json
    def _held(text: str) -> bool:
        _segments, _calls, cut = claude_server.stream_window(text, 0)
        return cut < len(text)

    assert _held("просто текст") is False
    assert _held("до <tool") is True
    assert _held("до ```") is True
    assert _held("до ```json") is True
    assert _held('до ```json\n{"name":') is True
    assert _held("до ```python\ndef fib(n):\n    return n") is False
    assert _held("```bash\nls -la\n```\nГотово!") is False

    # 2. Unknown foreign model names are cached after the first miss
    seen_models: list[str | None] = []

    class FakePrimary:
        name = "warm"

        def answer(self, prompt: str, *, model: str | None = None):
            del prompt
            seen_models.append(model)
            if model == "claude-sonnet-4-5-20250929":
                raise RuntimeError("Model 'claude-sonnet-4-5-20250929' was not found")
            yield "ok"

        def close(self) -> None:
            return None

    engine = claude_server.ArenaEngine(engine="echo", log=lambda _: None)
    engine._primary = FakePrimary()
    engine._fallback = None
    assert "".join(engine.answer("turn 1", model="claude-sonnet-4-5-20250929")) == "ok"
    assert "".join(engine.answer("turn 2", model="claude-sonnet-4-5-20250929")) == "ok"
    assert "".join(engine.answer("turn 3", model="claude-sonnet-4-5-20250929")) == "ok"
    # Turn 1 falls back once; turns 2 and 3 go straight to None with 0 retries!
    assert seen_models == ["claude-sonnet-4-5-20250929", None, None, None]

    # 3. Live SSE streaming: short chunks and ```python fences stream immediately
    # before the generator finishes (zero 64-char stall and zero fence freeze!).
    class GatedCodeEngine:
        engine = "custom"
        tools_mode = "emulate"
        verbose = False

        def answer(self, prompt: str, *, model: str | None = None):
            del prompt, model
            yield "Hi!"
            yield "\n```"
            yield "python\ndef f():\n    return 42\n"
            # Wait briefly so the test client can observe already-flushed SSE deltas
            # BEFORE the generator finishes!
            time.sleep(0.12)
            yield "```\n"

        def close(self) -> None:
            return None

    server = claude_server.create_server(port=0, engine=GatedCodeEngine())
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        arrival_times: list[tuple[float, str]] = []
        started = time.perf_counter()
        with httpx.Client(timeout=5.0) as http:
            with http.stream(
                "POST",
                f"http://{host}:{port}/v1/messages",
                json={
                    "model": "arena-auto",
                    "max_tokens": 256,
                    "stream": True,
                    "tools": [
                        {
                            "name": "Bash",
                            "description": "run",
                            "input_schema": {"type": "object", "properties": {}},
                        }
                    ],
                    "messages": [{"role": "user", "content": "code"}],
                },
            ) as resp:
                assert resp.status_code == 200
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta":
                            txt = data.get("delta", {}).get("text", "")
                            arrival_times.append(
                                ((time.perf_counter() - started) * 1000.0, txt)
                            )
        # Both "Hi!" (< 64 chars!) AND the ```python body arrived BEFORE the
        # 120 ms generator sleep!
        early_text = "".join(t for ms, t in arrival_times if ms < 100.0)
        assert "Hi!" in early_text
        assert "def f():" in early_text
    finally:
        server.shutdown()
        server.server_close()


def test_dynamic_tail_holdback_protects_split_tool_tags_and_stop_sequences() -> None:
    class SplitMarkerEngine:
        engine = "custom"
        tools_mode = "emulate"
        verbose = False

        def answer(self, prompt: str, *, model: str | None = None):
            del model
            if "STOP_CASE" in prompt:
                yield "начало ST"
                yield "OP хвост"
            else:
                yield "преамбула <to"
                yield 'ol_call>{"name": "Bash", "arguments": {"command": "pwd"}}</tool_call>'

        def close(self) -> None:
            return None

    server = claude_server.create_server(port=0, engine=SplitMarkerEngine())
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        with httpx.Client(timeout=5.0) as http:
            # Split <tool_call> across two deltas: "<to" must NOT leak as text!
            r1 = http.post(
                f"http://{host}:{port}/v1/messages",
                json={
                    "model": "arena-auto",
                    "max_tokens": 128,
                    "stream": True,
                    "tools": [
                        {
                            "name": "Bash",
                            "description": "run",
                            "input_schema": {"type": "object", "properties": {}},
                        }
                    ],
                    "messages": [{"role": "user", "content": "TOOL_CASE"}],
                },
            )
            text_deltas_1: list[str] = []
            tool_uses_1: list[str] = []
            for line in r1.text.splitlines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    if ev.get("type") == "content_block_delta":
                        delta = ev.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text_deltas_1.append(delta.get("text", ""))
                    elif ev.get("type") == "content_block_start":
                        cb = ev.get("content_block", {})
                        if cb.get("type") == "tool_use":
                            tool_uses_1.append(cb.get("name", ""))
            assert "".join(text_deltas_1) == "преамбула "
            assert tool_uses_1 == ["Bash"]

            # Split stop_sequence "STOP" across two deltas ("ST" + "OP"):
            r2 = http.post(
                f"http://{host}:{port}/v1/messages",
                json={
                    "model": "arena-auto",
                    "max_tokens": 128,
                    "stream": True,
                    "stop_sequences": ["STOP"],
                    "messages": [{"role": "user", "content": "STOP_CASE"}],
                },
            )
            text_deltas_2: list[str] = []
            for line in r2.text.splitlines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    if ev.get("type") == "content_block_delta":
                        delta = ev.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text_deltas_2.append(delta.get("text", ""))
            assert "".join(text_deltas_2) == "начало "
    finally:
        server.shutdown()
        server.server_close()


def test_fast_prompt_main_skips_redundant_ping_on_live_daemon_and_flags_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # CLI & env parsing across auto, warm, and claude shim
    parser = auto_module.build_arg_parser()
    assert auto_module._options_from_args(parser.parse_args([])).fast_model is True
    assert (
        auto_module._options_from_args(
            parser.parse_args(["--no-fast-model"])
        ).fast_model
        is False
    )
    monkeypatch.setenv("ARENA_FAST_MODEL", "0")
    assert auto_module.AutoOptions().fast_model is False
    assert (
        auto_module._options_from_args(
            parser.parse_args(["--fast-model"])
        ).fast_model
        is True
    )
    monkeypatch.delenv("ARENA_FAST_MODEL", raising=False)

    scanned = warm.scan_argv(["--prompt", "hi", "--no-fast-model"])
    assert scanned.fast_model is False
    assert "--no-fast-model" in scanned.rest

    # Live _WarmServer: fast_prompt_main answers in ONE TCP connection (0 pings!)
    state_file = tmp_path / "warm-daemon.json"
    ping_calls = 0
    real_ping = warm.ping

    def counting_ping(*a: Any, **kw: Any) -> bool:
        nonlocal ping_calls
        ping_calls += 1
        return real_ping(*a, **kw)

    monkeypatch.setattr(warm, "ping", counting_ping)

    class LiveSession:
        async def ask(self, prompt: str, model: str | None = None):
            yield f"fast:{prompt}:{model}"

        async def aclose(self) -> None:
            return None

    async def _run() -> None:
        server = warm._WarmServer(
            LiveSession(),
            state_file=state_file,
            token="live-tok",
        )
        await server.start()
        try:
            rc = await asyncio.to_thread(
                warm.fast_prompt_main,
                [
                    "--prompt",
                    "pingless",
                    "--no-fast-model",
                    "--warm-state",
                    str(state_file),
                ],
            )
            assert rc == 0
        finally:
            await server._shutdown()

    asyncio.run(_run())
    assert ping_calls == 0
    assert "answer            : fast:pingless:max" in capsys.readouterr().out


def test_composer_remount_after_first_keystroke_recovers_without_30s_timeout() -> None:
    """Reproduces the >32-char prompt remount race (`Locator.fill: Timeout 30000ms exceeded`)."""
    long_prompt = "привет! как дела? что думаешь насчет esp32?"
    assert len(long_prompt) > 32

    state = {
        "composer_marked": False,
        "filled_value": "",
        "fill_attempts": 0,
        "find_calls": 0,
        "send_clicks": 0,
    }

    class RemountingKeyboard:
        async def type(self, text: str, delay: int = 0) -> None:
            del text, delay
            # Typing the first character triggers a React remount that wipes
            # the [data-arena-composer="1"] attribute off the old <textarea>!
            state["composer_marked"] = False

    class RemountingLocator:
        def __init__(self, selector: str) -> None:
            self.selector = selector

        @property
        def first(self) -> RemountingLocator:
            return self

        async def focus(self, *, timeout: int = 30000) -> None:
            if "data-arena-composer" in self.selector and not state["composer_marked"]:
                await asyncio.sleep(timeout / 1000.0)
                raise TimeoutError("Locator.focus: Timeout exceeded")

        async def click(self, *, timeout: int = 30000, force: bool = False) -> None:
            del force
            if "data-arena-send" in self.selector:
                state["send_clicks"] += 1
                return
            if "data-arena-composer" in self.selector and not state["composer_marked"]:
                await asyncio.sleep(timeout / 1000.0)
                raise TimeoutError("Locator.click: Timeout exceeded")

        async def fill(self, value: str, *, timeout: int = 30000) -> None:
            state["fill_attempts"] += 1
            if not state["composer_marked"]:
                # Simulate real Playwright waiting up to min(timeout, 50ms in test)
                # and raising TimeoutError when [data-arena-composer="1"] is gone!
                assert timeout <= 2500, f"must not use 30000ms default timeout (got {timeout})"
                raise TimeoutError(
                    'Locator.fill: Timeout exceeded waiting for locator("[data-arena-composer=\\"1\\"]").first'
                )
            state["filled_value"] = value

        async def press(self, key: str, *, timeout: int = 30000) -> None:
            del key, timeout

    class RemountingPage:
        keyboard = RemountingKeyboard()

        def locator(self, selector: str) -> RemountingLocator:
            return RemountingLocator(selector)

        async def wait_for_timeout(self, ms: int) -> None:
            del ms

        async def evaluate(self, script: str) -> Any:
            if "arena_cleanup_marks" in script:
                state["composer_marked"] = False
                return None
            if "composer_nonempty" in script:
                return bool(state["filled_value"])
            if "data-arena-composer" in script and "data-arena-send" not in script and "arena_fill_composer_fallback" not in script:
                state["find_calls"] += 1
                state["composer_marked"] = True
                return {"found": True, "tag": "textarea", "x": 400, "y": 700}
            if "data-arena-send" in script:
                return {"found": True, "confidence": "explicit", "x": 650, "y": 700}
            return None

    started = time.perf_counter()
    ok = asyncio.run(
        direct_page._submit_prompt_via_composer(RemountingPage(), long_prompt)
    )
    elapsed = time.perf_counter() - started
    assert ok is True
    assert elapsed < 1.0
    assert state["fill_attempts"] == 2
    assert state["find_calls"] == 2
    assert state["filled_value"] == long_prompt
    assert state["send_clicks"] == 1

    # Also verify fallback to native JS React _valueTracker fill when Locator.fill
    # ALWAYS fails (e.g. covered by floating portal / actionability block).
    js_filled: list[str] = []

    class BlockedFillLocator(RemountingLocator):
        async def fill(self, value: str, *, timeout: int = 30000) -> None:
            del value, timeout
            raise TimeoutError("Locator.fill blocked by overlay")

    class BlockedFillPage(RemountingPage):
        def locator(self, selector: str) -> BlockedFillLocator:
            return BlockedFillLocator(selector)

        async def evaluate(self, script: str) -> Any:
            if "arena_fill_composer_fallback" in script:
                js_filled.append(script)
                return True
            return await super().evaluate(script)

    ok_fallback = asyncio.run(
        direct_page._submit_prompt_via_composer(BlockedFillPage(), long_prompt)
    )
    assert ok_fallback is True
    assert len(js_filled) == 1
    assert json.dumps(long_prompt) in js_filled[0]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_v8_composer_hydration_observer_and_react_value_tracker_fallback() -> None:
    """Executes _FIND_COMPOSER_JS (hydration + MutationObserver) and _build_js_fill_composer in V8."""
    prompt = "привет! как дела? что думаешь насчет esp32?"
    find_js = direct_page._FIND_COMPOSER_JS
    fill_js = direct_page._build_js_fill_composer(prompt)
    cleanup_js = direct_page._CLEANUP_ARENA_DOM_MARKS_JS

    harness = f"""
    let currentComposer = null;
    let observerCallback = null;
    let observerDisconnected = false;

    function makeTextarea(label, hydrated) {{
      const attrs = new Map();
      const events = [];
      let rawVal = "";
      const el = {{
        id: "",
        tagName: "TEXTAREA",
        label,
        disabled: false,
        readOnly: false,
        isConnected: true,
        events,
        getAttribute: (k) => attrs.has(k) ? attrs.get(k) : null,
        setAttribute: (k, v) => {{ attrs.set(k, String(v)); }},
        removeAttribute: (k) => {{ attrs.delete(k); }},
        closest: () => ({{ tagName: "FORM" }}),
        getBoundingClientRect: () => ({{ left: 100, top: 600, width: 700, height: 48 }}),
        focus: () => {{ el.focused = true; }},
        dispatchEvent: (ev) => {{ events.push(ev.type); return true; }},
        _valueTracker: {{
          tracked: "",
          setValue(v) {{ this.tracked = v; }}
        }},
      }};
      Object.defineProperty(el, "value", {{
        get() {{ return rawVal; }},
        set(v) {{ rawVal = String(v); }},
        configurable: true,
      }});
      if (hydrated) {{
        el.__reactProps$abc123 = {{ onChange: () => {{}} }};
      }}
      return el;
    }}

    // Start with an UNHYDRATED SSR textarea on a Next.js page; hydrate it after 35ms!
    const ssrTextarea = makeTextarea("ssr", false);
    currentComposer = ssrTextarea;
    let hadAttrBeforeHydration = false;
    setTimeout(() => {{
      hadAttrBeforeHydration = ssrTextarea.getAttribute("data-arena-composer") !== null;
      ssrTextarea.__reactProps$hydrated = {{}};
    }}, 35);

    global.MutationObserver = class {{
      constructor(cb) {{ observerCallback = cb; }}
      observe() {{}}
      disconnect() {{ observerDisconnected = true; }}
    }};
    global.InputEvent = class {{
      constructor(type, init) {{ this.type = type; Object.assign(this, init || {{}}); }}
    }};
    global.Event = class {{
      constructor(type, init) {{ this.type = type; Object.assign(this, init || {{}}); }}
    }};
    global.KeyboardEvent = class {{
      constructor(type, init) {{ this.type = type; Object.assign(this, init || {{}}); }}
    }};
    global.HTMLTextAreaElement = function() {{}};
    Object.defineProperty(global.HTMLTextAreaElement.prototype, "value", {{
      set(v) {{ this._protoSet = String(v); this.value = String(v); }},
      configurable: true,
    }});

    global.window = {{
      getComputedStyle: () => ({{ display: "block", visibility: "visible", opacity: "1" }}),
      HTMLTextAreaElement: global.HTMLTextAreaElement,
    }};
    global.document = {{
      body: {{ dispatchEvent: () => true }},
      documentElement: {{}},
      scripts: [],
      querySelector: (sel) => {{
        if (sel.includes("/_next/static/")) return {{ src: "/_next/static/chunks/main.js" }};
        if (sel === '[data-arena-composer="1"]') {{
          return (currentComposer && currentComposer.getAttribute("data-arena-composer") === "1")
            ? currentComposer
            : null;
        }}
        return null;
      }},
      querySelectorAll: (sel) => {{
        if (sel.includes("button")) return [];
        if (sel.includes("data-arena-composer")) {{
          return (currentComposer && currentComposer.getAttribute("data-arena-composer") !== null)
            ? [currentComposer]
            : [];
        }}
        if (sel.includes("textarea")) return currentComposer ? [currentComposer] : [];
        return [];
      }},
    }};

    (async () => {{
      const findFn = {find_js};
      const found = await findFn();
      // Simulate React remounting <textarea> when first character is typed!
      ssrTextarea.isConnected = false;
      const remountedTextarea = makeTextarea("remounted", true);
      remountedTextarea.value = "п";
      remountedTextarea._valueTracker.setValue("п");
      currentComposer = remountedTextarea;
      if (observerCallback) observerCallback();
      const autoRestamped = remountedTextarea.getAttribute("data-arena-composer") === "1";

      // Now run _build_js_fill_composer fallback on the remounted textarea!
      remountedTextarea.removeAttribute("data-arena-composer");
      const fillFn = {fill_js};
      const fillOk = fillFn();

      const cleanupFn = {cleanup_js};
      cleanupFn();

      console.log(JSON.stringify({{
        found: found.found,
        hadAttrBeforeHydration,
        autoRestamped,
        fillOk,
        finalValue: remountedTextarea.value,
        trackerPrev: remountedTextarea._valueTracker.tracked,
        events: remountedTextarea.events,
        observerDisconnected,
        cleanedUp: remountedTextarea.getAttribute("data-arena-composer") === null,
      }}));
    }})();
    """
    proc = subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    out = json.loads(proc.stdout.strip())
    assert out["found"] is True
    assert out["hadAttrBeforeHydration"] is False
    assert out["autoRestamped"] is True
    assert out["fillOk"] is True
    assert out["finalValue"] == prompt
    assert out["trackerPrev"] == "п"
    assert "input" in out["events"]
    assert "change" in out["events"]
    assert out["observerDisconnected"] is True
    assert out["cleanedUp"] is True


def test_warm_forwards_archive_after_answer_per_request_and_bumps_protocol(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert warm.PROTOCOL_VERSION >= 2
    scanned = warm.scan_argv(
        [
            "--prompt",
            "привет! как дела? что думаешь насчет esp32?",
            "--archive-after-answer",
        ]
    )
    assert scanned.archive_after_answer is True
    assert "--archive-after-answer" in scanned.rest

    # Stale protocol=1 daemon state is never optimistically reused
    stale_path = tmp_path / "stale-daemon.json"
    warm.DaemonState(
        pid=os.getpid(),
        port=65500,
        token="old",
        protocol=1,
        api_version=warm.api_version(),
        base_url=warm.DEFAULT_BASE_URL,
    ).write(stale_path)
    stale_parsed = warm.scan_argv(
        ["--prompt", "hi", "--warm-state", str(stale_path)]
    )
    assert warm._optimistic_live_state(stale_parsed) is None

    # Live protocol=2 daemon receives per-request archive=True even if started earlier
    seen_archive: list[bool | None] = []

    class ArchiveAwareSession:
        _accepts_config = True

        async def ask(
            self,
            prompt: str,
            model: str | None = None,
            *,
            config: dict[str, Any] | None = None,
            archive: bool | None = None,
        ):
            del model, config
            seen_archive.append(archive)
            yield f"ok:{len(prompt)}"

        async def aclose(self) -> None:
            return None

    live_path = tmp_path / "live-daemon.json"

    async def _run() -> None:
        server = warm._WarmServer(
            ArchiveAwareSession(),
            state_file=live_path,
            token="tok2",
        )
        await server.start()
        try:
            rc = await asyncio.to_thread(
                warm.fast_prompt_main,
                [
                    "--prompt",
                    "привет! как дела? что думаешь насчет esp32?",
                    "--archive-after-answer",
                    "--warm-state",
                    str(live_path),
                ],
            )
            assert rc == 0
        finally:
            await server._shutdown()

    asyncio.run(_run())
    assert seen_archive == [True]
    assert "answer            : ok:43" in capsys.readouterr().out


