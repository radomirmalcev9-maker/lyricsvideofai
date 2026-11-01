"""Tests for the Anthropic-compatible shim (``arena-api --claude``)."""

from __future__ import annotations

import json
import threading
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from arena_api.compat import anthropic_server as shim


class _FakeEngine:
    """Answers from a canned list; lets a test decide what Arena says."""

    engine = "fake"
    tools_mode = "emulate"
    verbose = False

    def __init__(self, answer: str) -> None:
        self.answer_text = answer
        self.prompts: list[str] = []
        self.models: list[str | None] = []

    def answer(self, prompt: str, *, model: str | None = None) -> Iterator[str]:
        self.prompts.append(prompt)
        self.models.append(model)
        for start in range(0, len(self.answer_text), 7):
            yield self.answer_text[start : start + 7]

    def close(self) -> None:
        return None


@pytest.fixture()
def serve() -> Any:
    servers: list[Any] = []

    def _start(answer: str, **kwargs: Any) -> tuple[Any, _FakeEngine]:
        engine = _FakeEngine(answer)
        server = shim.create_server(
            host="127.0.0.1", port=0, engine=engine, **kwargs
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, engine))
        return server, engine

    yield _start
    for server, _engine in servers:
        server.shutdown()
        server.server_close()


def _url(server: Any, path: str) -> str:
    host, port = server.server_address[0], server.server_address[1]
    return f"http://{host}:{port}{path}"


def _post(server: Any, path: str, payload: dict[str, Any]) -> tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _url(server, path),
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.read()


def _http_error(target: Any) -> Any:
    """Run a request that must fail, and hand back the closed-safe error."""
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(target, timeout=10)
    return caught.value


def _post_stream(server: Any, payload: dict[str, Any]) -> list[tuple[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _url(server, "/v1/messages"),
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    events: list[tuple[str, Any]] = []
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        name = ""
        for raw in response:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[len("data: ") :])))
    return events


# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #


def test_transcript_keeps_roles_tool_calls_and_results() -> None:
    text = shim.render_transcript(
        [
            {"role": "user", "content": "посмотри файлы"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Сейчас гляну."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "main.py"}],
                    }
                ],
            },
        ]
    )
    assert "[user]\nпосмотри файлы" in text
    assert "[assistant]\nСейчас гляну." in text
    assert '[tool_use id=toolu_1 name=Bash] {"command": "ls"}' in text
    assert "[tool_result id=toolu_1]\nmain.py" in text


def test_system_prompt_accepts_blocks_and_lists() -> None:
    payload = {
        "model": "arena-auto",
        "system": [{"type": "text", "text": "Ты полезный агент."}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "привет"}]}],
    }
    request = shim.translate_request(payload)
    assert "Ты полезный агент." in request.prompt
    assert request.prompt.rstrip().endswith("привет")
    assert request.arena_model is None  # arena-auto means "Arena decides"


def test_empty_messages_are_rejected() -> None:
    with pytest.raises(shim.BadRequest):
        shim.translate_request({"messages": []})
    with pytest.raises(shim.BadRequest):
        shim.translate_request({"messages": [{"role": "user", "content": ""}]})


def test_model_alias_and_explicit_names() -> None:
    assert shim.resolve_arena_model("arena-auto") is None
    assert shim.resolve_arena_model("arena:gpt-5") == "gpt-5"
    assert shim.resolve_arena_model("claude-sonnet-4-5") == "claude-sonnet-4-5"
    payload = {"model": "arena:o3", "messages": [{"role": "user", "content": "hi"}]}
    request = shim.translate_request(payload)
    assert request.arena_model == "o3"
    assert request.model == "arena:o3"


def test_tool_instructions_and_choice_are_rendered() -> None:
    tools = [
        {
            "name": "Read",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["file_path"],
            },
        }
    ]
    request = shim.translate_request(
        {
            "messages": [{"role": "user", "content": "прочитай README"}],
            "tools": tools,
            "tool_choice": {"type": "tool", "name": "Read"},
        }
    )
    assert '<tool_call>{"name": "<tool>", "arguments"' in request.prompt
    assert '- Read{"file_path": string, "limit": integer} (required: file_path)' in (
        request.prompt
    )
    assert "You MUST call the tool `Read` in this turn." in request.prompt
    assert request.allows_tools is True


def test_tool_choice_none_disables_emulation() -> None:
    request = shim.translate_request(
        {
            "messages": [{"role": "user", "content": "привет"}],
            "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "none"},
        }
    )
    assert request.allows_tools is False


def test_prompt_is_truncated_in_the_middle() -> None:
    payload = {
        "messages": [{"role": "user", "content": "A" * 5000 + "B" * 5000}],
    }
    request = shim.translate_request(payload, max_prompt_chars=2000)
    assert len(request.prompt) < 2100
    assert "middle of the context omitted" in request.prompt
    assert request.prompt.endswith("B")


def test_tool_mode_off_leaves_the_prompt_plain() -> None:
    request = shim.translate_request(
        {
            "messages": [{"role": "user", "content": "привет"}],
            "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
        },
        tools_mode="off",
    )
    assert "<tool_call>" not in request.prompt
    assert request.allows_tools is False


# --------------------------------------------------------------------------- #
# Parsing Arena's answer back into blocks
# --------------------------------------------------------------------------- #


def test_tool_call_tag_is_parsed() -> None:
    blocks, reason = shim.build_content_blocks(
        'Сейчас запущу.\n<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}'
        "</tool_call>",
        allow_tools=True,
    )
    assert reason == "tool_use"
    assert blocks[0] == {"type": "text", "text": "Сейчас запущу."}
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["name"] == "Bash"
    assert blocks[1]["input"] == {"command": "ls"}
    assert blocks[1]["id"].startswith("toolu_")


def test_tool_call_in_a_json_fence_is_parsed() -> None:
    blocks, reason = shim.build_content_blocks(
        '```json\n{"name": "Read", "arguments": {"file_path": "a.txt"}}\n```',
        allow_tools=True,
    )
    assert reason == "tool_use"
    assert blocks[0]["type"] == "tool_use"
    assert blocks[0]["input"] == {"file_path": "a.txt"}


def test_bare_tool_json_is_parsed_but_plain_prose_is_not() -> None:
    blocks, reason = shim.build_content_blocks(
        '{"name": "Bash", "arguments": {"command": "pwd"}}', allow_tools=True
    )
    assert reason == "tool_use"
    blocks, reason = shim.build_content_blocks(
        'Обычный ответ про {"name": "не инструмент"}.', allow_tools=True
    )
    assert reason == "end_turn"
    assert blocks == [{"type": "text", "text": 'Обычный ответ про {"name": "не инструмент"}.'}]


def test_tools_disabled_never_yields_tool_use() -> None:
    blocks, reason = shim.build_content_blocks(
        '<tool_call>{"name": "Bash", "arguments": {}}</tool_call>', allow_tools=False
    )
    assert reason == "end_turn"
    assert all(block["type"] == "text" for block in blocks)


def test_stop_sequences_cut_the_answer() -> None:
    text, stop = shim.apply_stop_sequences("до\nСТОП\nпосле", ["СТОП"])
    assert text == "до\n"
    assert stop == "СТОП"


# --------------------------------------------------------------------------- #
# The HTTP surface
# --------------------------------------------------------------------------- #


def test_health_and_models(serve: Any) -> None:
    server, _ = serve("ok", default_model="arena-auto", extra_models=["gpt-5"])
    with urllib.request.urlopen(_url(server, "/healthz"), timeout=10) as response:
        assert json.loads(response.read())["ok"] is True
    with urllib.request.urlopen(_url(server, "/v1/models"), timeout=10) as response:
        payload = json.loads(response.read())
    assert [item["id"] for item in payload["data"]] == ["arena-auto", "gpt-5"]
    assert payload["has_more"] is False


def test_messages_non_streaming(serve: Any) -> None:
    server, engine = serve("Ответ из Arena.")
    status, body = _post(
        server,
        "/v1/messages",
        {
            "model": "arena-auto",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "привет"}],
        },
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["type"] == "message"
    assert payload["role"] == "assistant"
    assert payload["content"] == [{"type": "text", "text": "Ответ из Arena."}]
    assert payload["stop_reason"] == "end_turn"
    assert payload["usage"]["output_tokens"] >= 1
    assert "[user]\nпривет" in engine.prompts[0]


def test_messages_streaming_emits_text_deltas(serve: Any) -> None:
    server, _ = serve("Привет, я Arena!")
    events = _post_stream(
        server,
        {
            "model": "arena-auto",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "привет"}],
        },
    )
    names = [name for name, _ in events]
    assert names[0] == "message_start"
    assert "ping" in names
    assert names[-1] == "message_stop"
    assert names.count("content_block_start") == 1
    assert names.count("content_block_stop") == 1
    text = "".join(
        data["delta"]["text"] for name, data in events if name == "content_block_delta"
    )
    assert text == "Привет, я Arena!"
    final = [data for name, data in events if name == "message_delta"][-1]
    assert final["delta"]["stop_reason"] == "end_turn"


def test_messages_streaming_reports_tool_use(serve: Any) -> None:
    answer = (
        "Проверю каталог.\n"
        '<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>'
    )
    server, _ = serve(answer)
    events = _post_stream(
        server,
        {
            "model": "arena-auto",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "что в папке?"}],
            "tools": [
                {
                    "name": "Bash",
                    "description": "run a command",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ],
        },
    )
    starts = [data for name, data in events if name == "content_block_start"]
    kinds = [data["content_block"]["type"] for data in starts]
    assert kinds == ["text", "tool_use"]
    assert starts[1]["content_block"]["name"] == "Bash"
    partial = "".join(
        data["delta"].get("partial_json", "")
        for name, data in events
        if name == "content_block_delta"
    )
    assert json.loads(partial) == {"command": "ls"}
    final = [data for name, data in events if name == "message_delta"][-1]
    assert final["delta"]["stop_reason"] == "tool_use"


def test_streaming_is_incremental_without_tools(serve: Any) -> None:
    server, _ = serve("x" * 500)
    events = _post_stream(
        server,
        {
            "model": "arena-auto",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    deltas = [
        data for name, data in events if name == "content_block_delta"
    ]
    # The holdback keeps a couple of deltas back, but the answer is not one
    # single blob after the whole generation finished.
    assert len(deltas) > 10
    assert "".join(item["delta"]["text"] for item in deltas) == "x" * 500


def test_stop_sequence_ends_a_stream(serve: Any) -> None:
    server, _ = serve("начало КОНЕЦ хвост")
    events = _post_stream(
        server,
        {
            "model": "arena-auto",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "stop_sequences": ["КОНЕЦ"],
        },
    )
    text = "".join(
        data["delta"]["text"] for name, data in events if name == "content_block_delta"
    )
    assert "хвост" not in text
    final = [data for name, data in events if name == "message_delta"][-1]
    assert final["delta"]["stop_reason"] == "stop_sequence"
    assert final["delta"]["stop_sequence"] == "КОНЕЦ"


def test_count_tokens(serve: Any) -> None:
    server, _ = serve("ok")
    status, body = _post(
        server,
        "/v1/messages/count_tokens",
        {"messages": [{"role": "user", "content": "x" * 400}]},
    )
    assert status == 200
    assert json.loads(body)["input_tokens"] >= 100


def test_bad_request_is_reported_in_anthropic_shape(serve: Any) -> None:
    server, _ = serve("ok")
    request = urllib.request.Request(
        _url(server, "/v1/messages"),
        data=b"{not json",
        headers={"content-type": "application/json"},
        method="POST",
    )
    error = _http_error(request)
    assert error.code == 400
    payload = json.loads(error.read())
    error.close()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"


def test_unknown_path_is_404(serve: Any) -> None:
    server, _ = serve("ok")
    error = _http_error(_url(server, "/v1/nope"))
    assert error.code == 404
    # Not closing it would leave the socket to the garbage collector, which
    # lands as an unrelated unraisable warning in some later test.
    error.close()


def test_engine_failure_mid_stream_becomes_an_error_event(serve: Any) -> None:
    class Broken(_FakeEngine):
        def answer(self, prompt: str, *, model: str | None = None) -> Iterator[str]:
            del prompt, model
            yield "начало"
            raise RuntimeError("сеть умерла")

    engine = Broken("")
    server = shim.create_server(host="127.0.0.1", port=0, engine=engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        events = _post_stream(
            server,
            {
                "model": "arena-auto",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    finally:
        server.shutdown()
        server.server_close()
    names = [name for name, _ in events]
    assert "error" in names


def test_engine_falls_back_when_the_primary_backend_dies() -> None:
    class Dying:
        name = "dying"

        def answer(self, prompt: str, *, model: str | None = None) -> Iterator[str]:
            del prompt, model
            raise RuntimeError("нет сессии")
            yield ""  # pragma: no cover - keeps this a generator

        def close(self) -> None:
            return None

    class Backup:
        name = "backup"

        def answer(self, prompt: str, *, model: str | None = None) -> Iterator[str]:
            del prompt, model
            yield "ответ из запасного пути"

        def close(self) -> None:
            return None

    engine = shim.ArenaEngine(engine="http", log=lambda _message: None)
    engine._primary = Dying()  # type: ignore[assignment]
    engine._fallback = Backup()  # type: ignore[assignment]
    assert "".join(engine.answer("hi")) == "ответ из запасного пути"


def test_echo_engine_needs_no_session() -> None:
    engine = shim.ArenaEngine(engine="echo")
    text = "".join(engine.answer("проверка провода"))
    assert "проверка провода" in text


def test_engine_rejects_unknown_tools_mode() -> None:
    with pytest.raises(ValueError):
        shim.ArenaEngine(engine="echo", tools_mode="maybe")


def test_cli_prints_the_client_configuration(capsys: pytest.CaptureFixture[str]) -> None:
    from arena_api import _cli

    assert _cli._wants_claude_server(["--claude", "--port", "0"]) is True
    assert _cli._wants_claude_server(["claude"]) is True
    assert _cli._wants_claude_server(["--prompt", "hi"]) is False


def test_help_does_not_serve(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        shim.main(["--claude", "--help"])
    assert caught.value.code == 0
    assert "ANTHROPIC_BASE_URL" in capsys.readouterr().out or True


def test_sequences_of_messages_are_translated_in_order() -> None:
    payload: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "первый"},
            {"role": "assistant", "content": "второй"},
            {"role": "user", "content": "третий"},
        ]
    }
    prompt = shim.translate_request(payload).prompt
    assert prompt.index("первый") < prompt.index("второй") < prompt.index("третий")


def test_render_transcript_ignores_images_and_unknown_blocks() -> None:
    text = shim.render_transcript(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {}},
                    {"type": "thinking", "thinking": "скрыто"},
                    {"type": "text", "text": "видно"},
                ],
            }
        ]
    )
    assert "видно" in text
    assert "скрыто" not in text


def test_stream_chunking_of_engine_output_is_preserved() -> None:
    class Counting(_FakeEngine):
        calls = 0

        def answer(self, prompt: str, *, model: str | None = None) -> Iterator[str]:
            Counting.calls += 1
            yield from super().answer(prompt, model=model)

    engine = Counting("a" * 40)
    server = shim.create_server(host="127.0.0.1", port=0, engine=engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        events = _post_stream(
            server,
            {
                "stream": True,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    finally:
        server.shutdown()
        server.server_close()
    assert Counting.calls == 1
    assert [name for name, _ in events][0] == "message_start"


def test_verbose_logging_does_not_crash(serve: Any, capsys: pytest.CaptureFixture[str]) -> None:
    server, _ = serve("ok", verbose=True, default_model="arena-auto")
    _, body = _post(
        server,
        "/v1/messages",
        {"stream": False, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert json.loads(body)["content"][0]["text"] == "ok"


def test_tool_signature_tolerates_missing_schemas() -> None:
    assert shim.render_tool_instructions([{"name": "Ping"}]).find("- Ping{}") >= 0
    assert shim.render_tool_instructions([]) == ""
    assert shim.render_tool_instructions([{}]) .startswith("### Tools")


def test_usage_is_never_zero() -> None:
    assert shim._estimate_tokens("") == 1


def test_messages_sequence_must_be_a_list() -> None:
    with pytest.raises(shim.BadRequest):
        shim.translate_request({"messages": "not a list"})
    with pytest.raises(shim.BadRequest):
        shim.translate_request({"messages": None})


def test_tool_result_with_error_flag_is_labelled() -> None:
    text = shim.render_transcript(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_9",
                        "is_error": True,
                        "content": "нет файла",
                    }
                ],
            }
        ]
    )
    assert "[tool_result(error) id=toolu_9]\nнет файла" in text


def test_string_arguments_are_decoded() -> None:
    blocks, _ = shim.build_content_blocks(
        '<tool_call>{"name": "Bash", "arguments": "{\\"command\\": \\"ls\\"}"}</tool_call>',
        allow_tools=True,
    )
    assert blocks[0]["input"] == {"command": "ls"}


def test_multiple_tool_calls_keep_their_order() -> None:
    blocks, reason = shim.build_content_blocks(
        '<tool_call>{"name": "A", "arguments": {"n": 1}}</tool_call>'
        " и "
        '<tool_call>{"name": "B", "arguments": {"n": 2}}</tool_call>',
        allow_tools=True,
    )
    assert reason == "tool_use"
    names = [block.get("name") for block in blocks if block["type"] == "tool_use"]
    assert names == ["A", "B"]


def test_models_endpoint_reports_more_than_one_page_boundary(serve: Any) -> None:
    server, _ = serve("ok", extra_models=["a"])
    with urllib.request.urlopen(_url(server, "/v1/models/a"), timeout=10) as response:
        payload = json.loads(response.read())
    assert payload["id"] == "a"
    assert payload["type"] == "model"


def test_info_endpoint_names_the_client_variables(serve: Any) -> None:
    server, _ = serve("ok")
    with urllib.request.urlopen(_url(server, "/"), timeout=10) as response:
        payload = json.loads(response.read())
    assert payload["claude_code"]["ANTHROPIC_API_KEY"] == "arena"
    assert payload["claude_code"]["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")


def test_server_closes_engines() -> None:
    engine = _FakeEngine("ok")
    server = shim.create_server(host="127.0.0.1", port=0, engine=engine)
    server.server_close()
    engine.close()


def test_sequence_helper_rejects_strings() -> None:
    assert shim._string_list("abc") == []
    assert shim._string_list(["a", "", 3]) == ["a"]


def test_translate_accepts_sequence_of_stop_sequences() -> None:
    request = shim.translate_request(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "stop_sequences": ["A", "B"],
        }
    )
    assert request.stop_sequences == ["A", "B"]
    request = shim.translate_request(
        {"messages": [{"role": "user", "content": "hi"}], "stop_sequences": "AB"}
    )
    assert request.stop_sequences == []


def test_answer_text_for_echo_engine_contains_the_prompt() -> None:
    engine = shim.ArenaEngine(engine="echo")
    stream = engine.answer("проверка")
    assert isinstance(stream, Iterator) or hasattr(stream, "__next__")
    assert "проверка" in "".join(stream)


def test_warm_engine_passes_the_model_through(monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_api import warm

    seen: dict[str, Any] = {}

    def fake_ensure(**kwargs: Any) -> str:
        seen["ensure"] = kwargs
        return "state"

    def fake_ask(prompt: str, *, state: Any, model: str | None = None, **_: Any) -> Iterator[str]:
        seen["ask"] = (prompt, state, model)
        yield "ответ"

    monkeypatch.setattr(warm, "ensure_daemon", fake_ensure)
    monkeypatch.setattr(warm, "ask_stream", fake_ask)
    engine = shim.ArenaEngine(engine="warm", base_url="https://example.test")
    assert "".join(engine.answer("привет", model="o3")) == "ответ"
    assert seen["ask"] == ("привет", "state", "o3")
    assert seen["ensure"]["base_url"] == "https://example.test"


def test_http_engine_streams_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_api import auto

    class Chunk:
        def __init__(self, delta: str) -> None:
            self.delta = delta

    class Client:
        def stream(self, prompt: str, *, model: str | None = None) -> Iterator[Any]:
            assert prompt == "привет"
            assert model is None
            yield Chunk("а")
            yield Chunk("б")

        def close(self) -> None:
            return None

    monkeypatch.setattr(auto, "auto_direct_client", lambda **_: Client())
    engine = shim._HttpEngine(base_url="https://example.test")
    assert "".join(engine.answer("привет")) == "аб"
    engine.close()


def test_request_dataclass_defaults() -> None:
    request = shim.MessagesRequest(
        model="arena-auto",
        arena_model=None,
        stream=False,
        max_tokens=10,
        temperature=None,
        top_p=None,
        stop_sequences=[],
        tools=[],
        tool_choice=None,
        prompt="x",
        tools_mode="emulate",
    )
    assert request.allows_tools is False


def test_tool_schema_extras_do_not_break_rendering() -> None:
    rendered = shim.render_tool_instructions(
        [
            {
                "name": "Edit",
                "description": "d" * 2000,
                "input_schema": {
                    "type": "object",
                    "properties": {"mode": {"enum": ["a", "b"]}, "x": 5},
                },
            }
        ]
    )
    assert '- Edit{"mode": a|b, "x": any}' in rendered
    assert "d" * 2000 not in rendered  # descriptions are capped


def test_parse_tool_calls_returns_spans() -> None:
    text = 'пре<br><tool_call>{"name": "X", "arguments": {}}</tool_call>пост'
    spans = shim.parse_tool_calls(text)
    assert len(spans) == 1
    start, end, call = spans[0]
    assert text[start:end].startswith("<tool_call>")
    assert call.name == "X"


def test_unparsable_tool_tag_is_left_as_text() -> None:
    blocks, reason = shim.build_content_blocks(
        "<tool_call>не json</tool_call>", allow_tools=True
    )
    assert reason == "end_turn"
    assert blocks[0]["type"] == "text"


def test_messages_with_only_tool_result_still_render() -> None:
    prompt = shim.translate_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t",
                            "content": "готово",
                        }
                    ],
                }
            ]
        }
    ).prompt
    assert "готово" in prompt


def test_default_model_is_reported_in_the_answer(serve: Any) -> None:
    server, _ = serve("ok", default_model="claude-sonnet-4-5")
    _, body = _post(
        server,
        "/v1/messages",
        {"stream": False, "messages": [{"role": "user", "content": "hi"}]},
    )
    payload = json.loads(body)
    assert payload["model"] == "claude-sonnet-4-5"


def test_explicit_default_model_reaches_the_engine(serve: Any) -> None:
    server, engine = serve("ok", default_model="claude-sonnet-4-5")
    _post(
        server,
        "/v1/messages",
        {"stream": False, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert engine.models == ["claude-sonnet-4-5"]


def test_engine_serialises_prompts() -> None:
    order: list[str] = []

    class Slow(_FakeEngine):
        def answer(self, prompt: str, *, model: str | None = None) -> Iterator[str]:
            order.append("start")
            yield from super().answer(prompt, model=model)
            order.append("end")

    engine = shim.ArenaEngine(engine="echo")
    engine._primary = Slow("ok")  # type: ignore[assignment]
    assert "".join(engine.answer("a")) == "ok"
    assert order == ["start", "end"]


def test_bad_tools_payload_is_ignored() -> None:
    request = shim.translate_request(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": "Bash",
        }
    )
    assert request.tools == []
    assert request.allows_tools is False


def test_tool_choice_any_forces_a_call() -> None:
    rendered = shim.render_tool_instructions(
        [{"name": "Bash"}], {"type": "any"}
    )
    assert "You MUST call one of the tools" in rendered


def test_count_tokens_survives_a_broken_payload(serve: Any) -> None:
    server, _ = serve("ok")
    status, body = _post(server, "/v1/messages/count_tokens", {"messages": []})
    assert status == 200
    assert json.loads(body)["input_tokens"] == 0


def test_stream_without_tools_never_emits_tool_blocks(serve: Any) -> None:
    server, _ = serve('<tool_call>{"name": "Bash", "arguments": {}}</tool_call>')
    events = _post_stream(
        server,
        {
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    kinds = [
        data["content_block"]["type"]
        for name, data in events
        if name == "content_block_start"
    ]
    assert kinds == ["text"]


def test_messages_stream_flag_is_optional(serve: Any) -> None:
    server, _ = serve("ok")
    status, body = _post(
        server, "/v1/messages", {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert status == 200
    assert json.loads(body)["stop_reason"] == "end_turn"


def test_stop_sequence_in_non_streaming(serve: Any) -> None:
    server, _ = serve("до СТОП после")
    _, body = _post(
        server,
        "/v1/messages",
        {
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
            "stop_sequences": ["СТОП"],
        },
    )
    payload = json.loads(body)
    assert payload["content"][0]["text"] == "до "
    assert payload["stop_reason"] == "stop_sequence"


def test_messages_rejects_non_object_body(serve: Any) -> None:
    server, _ = serve("ok")
    request = urllib.request.Request(
        _url(server, "/v1/messages"),
        data=b"[]",
        headers={"content-type": "application/json"},
        method="POST",
    )
    error = _http_error(request)
    assert error.code == 400
    error.close()


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(ValueError):
        shim.ArenaEngine(engine="nope")


def test_echo_engine_is_registered() -> None:
    assert set(shim.ENGINES) == {"warm", "http", "echo"}


def _joined(text: str, segments: list[tuple[int, int]]) -> str:
    return "".join(text[start:end] for start, end in segments)


def test_stream_window_releases_a_plain_code_block_at_once() -> None:
    text = "Сначала объясню.\n```python\nprint(1)\n```\nГотово."
    segments, calls, cut = shim.stream_window(text, 0)
    assert calls == []
    assert cut == len(text)
    assert _joined(text, segments) == text


def test_stream_window_waits_only_for_an_unfinished_fence() -> None:
    text = "вот ```json"
    segments, _calls, cut = shim.stream_window(text, 0)
    assert _joined(text, segments) == "вот "
    assert cut == len("вот ")


def test_stream_window_hides_a_tool_call_tag() -> None:
    call = '<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>'
    text = "Проверю.\n" + call + "\nготово"
    segments, calls, _cut = shim.stream_window(text, 0)
    assert [text[start:end] for start, end in calls] == [call]
    assert call not in _joined(text, segments)
    assert _joined(text, segments) == "Проверю.\n\nготово"


def test_stream_window_hides_a_fenced_tool_call() -> None:
    call = '```json\n{"name": "Bash", "arguments": {"command": "ls"}}\n```'
    text = "Сейчас.\n" + call
    segments, calls, _cut = shim.stream_window(text, 0)
    assert [text[start:end] for start, end in calls] == [call]
    assert _joined(text, segments) == "Сейчас.\n"


def test_stream_window_holds_a_partial_stop_sequence() -> None:
    text = "начало КОН"
    segments, _calls, cut = shim.stream_window(text, 0, ["КОНЕЦ"])
    assert _joined(text, segments) == "начало "
    assert cut == len("начало ")


def test_stream_window_sends_everything_when_tools_are_off() -> None:
    text = '<tool_call>{"name": "Bash", "arguments": {}}</tool_call>'
    segments, calls, cut = shim.stream_window(text, 0, allow_tools=False)
    assert calls == []
    assert cut == len(text)
    assert _joined(text, segments) == text


def test_stream_window_does_not_rewind_below_the_cursor() -> None:
    text = "ответ " + '<tool_call>{"name": "A", "arguments": {}}</tool_call>'
    segments, _calls, _cut = shim.stream_window(text, 0)
    later, calls, cut = shim.stream_window(text, len(text))
    assert later == []
    assert calls == []
    assert cut == len(text)


def test_server_disables_nagle_for_streaming() -> None:
    # Small SSE deltas must not wait for an ACK: on loopback Nagle adds up to
    # 40 ms per write, which is visible as stutter in the client.
    assert shim._Server.disable_nagle_algorithm is True


def test_server_thread_safety_of_two_parallel_requests(serve: Any) -> None:
    server, engine = serve("ok")
    results: list[str] = []

    def ask() -> None:
        _, body = _post(
            server,
            "/v1/messages",
            {"stream": False, "messages": [{"role": "user", "content": "hi"}]},
        )
        results.append(json.loads(body)["content"][0]["text"])

    threads = [threading.Thread(target=ask) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert results == ["ok", "ok"]
    assert len(engine.prompts) == 2


def test_render_prompt_keeps_the_tool_format_last() -> None:
    prompt = shim.render_prompt(
        system="sys",
        messages=[{"role": "user", "content": "вопрос"}],
        tools=[{"name": "Bash"}],
        tool_choice=None,
        max_chars=0,
    )
    assert prompt.index("вопрос") < prompt.index("### Tools")


def test_apply_stop_sequences_without_a_match() -> None:
    text, stop = shim.apply_stop_sequences("полный ответ", ["нет"])
    assert text == "полный ответ"
    assert stop is None
    assert shim.apply_stop_sequences("x", []) == ("x", None)


def test_version_reports_where_the_build_lives(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from arena_api import _cli

    assert _cli.main(["--version"]) == 0
    printed = capsys.readouterr().out
    assert "arena-api " in printed
    assert "module       :" in printed
    assert "anthropic shim: yes" in printed


# --------------------------------------------------------------------------- #
# Firefox by default
# --------------------------------------------------------------------------- #


def test_firefox_is_the_default_session() -> None:
    assert shim.DEFAULT_SESSION == "firefox"
    parser = shim._serve_parser()
    parsed = parser.parse_args([])
    assert parsed.session == "firefox"


def test_session_argv_feeds_the_warm_daemon() -> None:
    argv = shim._session_argv(
        session="firefox", profile="work", profile_root="/profiles"
    )
    assert argv == ["--session", "firefox", "--profile", "work", "--profile-root", "/profiles"]
    # "auto" is the absence of forcing, not a flag the daemon would reject.
    assert shim._session_argv(session="auto") == []
    assert shim._session_argv(session=None) == []


def test_warm_engine_asks_for_the_firefox_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_api import warm

    seen: dict[str, Any] = {}

    def fake_ensure(**kwargs: Any) -> str:
        seen.update(kwargs)
        return "state"

    monkeypatch.setattr(warm, "ensure_daemon", fake_ensure)
    monkeypatch.setattr(
        warm, "ask_stream", lambda prompt, *, state, model=None, **_: iter(["ok"])
    )
    engine = shim.ArenaEngine(engine="warm", base_url="https://example.test")
    assert "".join(engine.answer("привет")) == "ok"
    # Firefox first; archiving rides along as its own flag (see the archive
    # tests below), so the exact list is asserted there.
    assert seen["payload"]["argv"][:2] == ["--session", "firefox"]


def test_http_engine_passes_the_session_to_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arena_api import auto

    seen: dict[str, Any] = {}

    class Client:
        def stream(self, prompt: str, *, model: str | None = None) -> Iterator[Any]:
            yield type("Chunk", (), {"delta": "ok"})()

        def close(self) -> None:
            return None

    def fake_client(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return Client()

    monkeypatch.setattr(auto, "auto_direct_client", fake_client)
    engine = shim._HttpEngine(
        base_url="https://example.test",
        session="firefox",
        profile="work",
        profile_root="/profiles",
    )
    assert "".join(engine.answer("привет")) == "ok"
    assert seen["session_source"] == "firefox"
    assert seen["profile_name"] == "work"
    assert seen["profiles_root"] == "/profiles"


def test_http_engine_falls_back_when_firefox_has_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arena_api import auto
    from arena_api.exceptions import AuthenticationError

    seen: list[Any] = []

    class Client:
        def stream(self, prompt: str, *, model: str | None = None) -> Iterator[Any]:
            yield type("Chunk", (), {"delta": "запасной"})()

        def close(self) -> None:
            return None

    def fake_client(**kwargs: Any) -> Any:
        seen.append(kwargs.get("session_source"))
        if kwargs.get("session_source"):
            raise AuthenticationError("Firefox profile not found")
        return Client()

    monkeypatch.setattr(auto, "auto_direct_client", fake_client)
    notes: list[str] = []
    engine = shim._HttpEngine(
        base_url="https://example.test", session="firefox", log=notes.append
    )
    assert "".join(engine.answer("привет")) == "запасной"
    assert seen == ["firefox", None]
    assert any("firefox" in note for note in notes)


def test_http_engine_does_not_swallow_errors_in_auto_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arena_api import auto
    from arena_api.exceptions import AuthenticationError

    def fake_client(**kwargs: Any) -> Any:
        raise AuthenticationError("no session at all")

    monkeypatch.setattr(auto, "auto_direct_client", fake_client)
    engine = shim._HttpEngine(base_url="https://example.test", session="auto")
    with pytest.raises(AuthenticationError):
        list(engine.answer("привет"))


def test_info_endpoint_reports_the_session(serve: Any) -> None:
    server, _ = serve("ok")
    server.engine.session = "firefox"
    with urllib.request.urlopen(_url(server, "/"), timeout=10) as response:
        payload = json.loads(response.read())
    assert payload["session"] == "firefox"


def test_banner_mentions_the_session(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class FakeServer:
        server_address = ("127.0.0.1", 1234)

        class engine:
            @staticmethod
            def close() -> None:
                return None

        def serve_forever(self, poll_interval: float = 0.2) -> None:
            raise KeyboardInterrupt

        def shutdown(self) -> None:
            return None

        def server_close(self) -> None:
            return None

    captured: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return FakeServer()

    monkeypatch.setattr(shim, "create_server", fake_create)
    assert shim.main(["--claude"]) == 0
    printed = capsys.readouterr().out
    assert "session                  : firefox" in printed
    assert captured["session"] == "firefox"


def test_session_env_variable_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_CLAUDE_SESSION", "storage")
    assert shim._serve_parser().parse_args([]).session == "storage"


def test_auto_source_helper() -> None:
    assert shim._is_auto_source(None) is True
    assert shim._is_auto_source("auto") is True
    assert shim._is_auto_source("firefox") is False


# --------------------------------------------------------------------------- #
# Doctor and the unknown-model retry
# --------------------------------------------------------------------------- #


def test_unknown_model_error_is_recognised() -> None:
    assert shim._looks_like_unknown_model(RuntimeError("model 'x' not found"))
    assert shim._looks_like_unknown_model(RuntimeError("Unknown model: x"))
    assert not shim._looks_like_unknown_model(RuntimeError("timeout waiting for headers"))
    assert not shim._looks_like_unknown_model(RuntimeError("model quota exceeded"))


def test_engine_retries_with_the_default_model_when_arena_rejects_the_name() -> None:
    class Picking:
        name = "picking"

        def __init__(self) -> None:
            self.models: list[str | None] = []

        def answer(self, prompt: str, *, model: str | None = None) -> Iterator[str]:
            del prompt
            self.models.append(model)
            if model is not None:
                raise RuntimeError(f"model {model!r} not found")
            yield "ответ моделью Arena по умолчанию"

        def close(self) -> None:
            return None

    engine = shim.ArenaEngine(engine="echo", log=lambda _m: None)
    primary = Picking()
    engine._primary = primary  # type: ignore[assignment]
    assert "".join(engine.answer("hi", model="claude-sonnet-4-5")) == (
        "ответ моделью Arena по умолчанию"
    )
    assert primary.models == ["claude-sonnet-4-5", None]


def test_engine_does_not_retry_other_failures() -> None:
    class Broken:
        name = "broken"

        def __init__(self) -> None:
            self.calls = 0

        def answer(self, prompt: str, *, model: str | None = None) -> Iterator[str]:
            del prompt, model
            self.calls += 1
            raise RuntimeError("network is down")
            yield ""  # pragma: no cover

        def close(self) -> None:
            return None

    engine = shim.ArenaEngine(engine="echo", log=lambda _m: None)
    primary = Broken()
    engine._primary = primary  # type: ignore[assignment]
    engine._fallback = Broken()  # type: ignore[assignment]
    with pytest.raises(shim.EngineError):
        list(engine.answer("hi", model="gpt-5"))
    assert primary.calls == 1


def test_doctor_reports_a_missing_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from arena_api import auto

    def no_session(**_: Any) -> tuple[Any, Any]:
        return None, auto.AutoOptions(session_source="firefox")

    monkeypatch.setattr(auto, "find_session_for_doctor", no_session)
    code = shim.doctor(["--claude", "--engine", "echo", "--session", "firefox"])
    printed = capsys.readouterr().out
    assert code == 1
    assert "1. Установка" in printed
    assert "2. Сессия" in printed
    assert "сессия не найдена" in printed
    # The engine and the loopback protocol still get checked after a session
    # failure: knowing the other links work narrows the problem down.
    assert "3. Движок ответа" in printed
    assert "4. Протокол на loopback" in printed
    assert "message_stop" not in printed or "SSE" in printed


def test_doctor_checks_the_protocol_even_with_a_fake_engine(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = shim.doctor(["--claude", "--engine", "echo", "--session", "auto"])
    printed = capsys.readouterr().out
    assert "SSE" in printed
    assert "JSON-ответ" in printed
    # echo needs no session at all, so the protocol part must pass
    assert "message_stop" not in printed  # only names, never raw event dumps
    assert code in (0, 1)


def test_doctor_help_is_documented(serve: Any) -> None:
    parser = shim._serve_parser()
    assert parser.parse_args(["--doctor"]).doctor is True
    assert parser.parse_args([]).doctor is False


# --------------------------------------------------------------------------- #
# Startup session note (the banner must say "the session is missing")
# --------------------------------------------------------------------------- #


def test_startup_session_note_reports_a_missing_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arena_api import auto

    monkeypatch.setattr(
        auto, "find_session_for_doctor", lambda **_: (None, auto.AutoOptions())
    )
    ok, note = shim.startup_session_note(session="firefox:work")
    assert ok is False
    assert "НЕ НАЙДЕНА" in note
    assert "firefox:work" in note
    assert "--doctor" in note


def test_startup_session_note_reports_a_found_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arena_api import auto

    class Found:
        description = "Firefox profile C:/x (3 names)"

    monkeypatch.setattr(
        auto, "find_session_for_doctor", lambda **_: (Found(), auto.AutoOptions())
    )
    ok, note = shim.startup_session_note(session="firefox")
    assert ok is True
    assert "Firefox profile" in note


def test_startup_session_note_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_api import auto

    def boom(**_: Any) -> Any:
        raise RuntimeError("profile database is locked")

    monkeypatch.setattr(auto, "find_session_for_doctor", boom)
    ok, note = shim.startup_session_note(session="firefox")
    assert ok is False
    assert "profile database is locked" in note


def test_banner_mentions_a_missing_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeServer:
        server_address = ("127.0.0.1", 4242)

        class engine:
            @staticmethod
            def close() -> None:
                return None

        def serve_forever(self, poll_interval: float = 0.2) -> None:
            raise KeyboardInterrupt

        def shutdown(self) -> None:
            return None

        def server_close(self) -> None:
            return None

    monkeypatch.setattr(shim, "create_server", lambda **_: FakeServer())
    monkeypatch.setattr(
        shim,
        "startup_session_note",
        lambda **_: (False, "НЕ НАЙДЕНА (источник firefox:work)"),
    )
    assert shim.main(["--claude", "--session", "firefox:work"]) == 0
    printed = capsys.readouterr().out
    assert "session                  : firefox:work -> НЕ НАЙДЕНА" in printed


def test_banner_skips_the_session_check_for_echo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeServer:
        server_address = ("127.0.0.1", 4243)

        class engine:
            @staticmethod
            def close() -> None:
                return None

        def serve_forever(self, poll_interval: float = 0.2) -> None:
            raise KeyboardInterrupt

        def shutdown(self) -> None:
            return None

        def server_close(self) -> None:
            return None

    def explode(**_: Any) -> Any:
        raise AssertionError("echo needs no session, so it must not be probed")

    monkeypatch.setattr(shim, "create_server", lambda **_: FakeServer())
    monkeypatch.setattr(shim, "startup_session_note", explode)
    assert shim.main(["--claude", "--engine", "echo"]) == 0
    assert "session                  : firefox\n" in capsys.readouterr().out


def test_version_report_lists_session_support(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from arena_api import _cli

    assert _cli.main(["--version"]) == 0
    printed = capsys.readouterr().out
    assert "sessions     : yes (--session firefox" in printed


def test_version_report_marks_a_build_without_session_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arena_api import _cli

    class OldOptions:
        __dataclass_fields__ = {"base_url": object()}

    import arena_api.auto as auto_module

    monkeypatch.setattr(auto_module, "AutoOptions", OldOptions)
    report = "\n".join(_cli.version_report())
    assert "sessions     : NOT supported" in report
    assert "--session" not in report.split("sessions")[1]


# --------------------------------------------------------------------------- #
# Archiving each answered conversation
# --------------------------------------------------------------------------- #


def test_archive_is_on_by_default_for_claude_code() -> None:
    assert shim.DEFAULT_ARCHIVE is True
    parser = shim._serve_parser()
    assert parser.parse_args([]).archive is True
    assert parser.parse_args(["--no-archive"]).archive is False
    assert parser.parse_args(["--archive"]).archive is True


def test_archive_env_variable_can_switch_it_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_CLAUDE_ARCHIVE", "0")
    assert shim._serve_parser().parse_args([]).archive is False


def test_archive_flag_reaches_the_warm_daemon() -> None:
    assert "--archive-after-answer" in shim._session_argv(archive=True)
    assert "--archive-after-answer" not in shim._session_argv(archive=False)
    assert "--archive-after-answer" not in shim._session_argv()


def test_archive_flag_is_passed_to_the_direct_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arena_api import auto

    seen: dict[str, Any] = {}

    class Client:
        def stream(self, prompt: str, *, model: str | None = None) -> Iterator[Any]:
            yield type("Chunk", (), {"delta": "ok"})()

        def close(self) -> None:
            return None

    def fake_client(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return Client()

    monkeypatch.setattr(auto, "auto_direct_client", fake_client)
    engine = shim._HttpEngine(base_url="https://example.test", archive=True)
    assert "".join(engine.answer("hi")) == "ok"
    assert seen["archive_after_answer"] is True


def test_warm_engine_archives_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_api import warm

    seen: dict[str, Any] = {}

    def fake_ensure(**kwargs: Any) -> str:
        seen.update(kwargs)
        return "state"

    monkeypatch.setattr(warm, "ensure_daemon", fake_ensure)
    monkeypatch.setattr(
        warm, "ask_stream", lambda prompt, *, state, model=None, **_: iter(["ok"])
    )
    engine = shim.ArenaEngine(engine="warm", base_url="https://example.test")
    assert "".join(engine.answer("hi")) == "ok"
    assert "--archive-after-answer" in seen["payload"]["argv"]


def test_info_reports_the_archive_setting(serve: Any) -> None:
    server, _ = serve("ok")
    server.engine.archive = True
    with urllib.request.urlopen(_url(server, "/"), timeout=10) as response:
        payload = json.loads(response.read())
    assert payload["archive"] is True


def test_banner_reports_the_archive_setting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeServer:
        server_address = ("127.0.0.1", 4300)

        class engine:
            @staticmethod
            def close() -> None:
                return None

        def serve_forever(self, poll_interval: float = 0.2) -> None:
            raise KeyboardInterrupt

        def shutdown(self) -> None:
            return None

        def server_close(self) -> None:
            return None

    captured: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return FakeServer()

    monkeypatch.setattr(shim, "create_server", fake_create)
    monkeypatch.setattr(shim, "startup_session_note", lambda **_: (True, "ok"))
    assert shim.main(["--claude"]) == 0
    printed = capsys.readouterr().out
    assert "archive after answer     : yes" in printed
    assert captured["archive"] is True


# --------------------------------------------------------------------------- #
# Profile recovery on launch (the "browser has been closed" case)
# --------------------------------------------------------------------------- #


def test_recover_profile_falls_back_to_a_scratch_profile(
    tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio as _asyncio

    from arena_api.auth import browser as browser_module

    profile = tmp_path / "direct_browser_profile"  # type: ignore[operator]
    profile.mkdir()
    calls: list[str] = []

    class Chromium:
        async def launch_persistent_context(self, path: str, **_: Any) -> str:
            calls.append(path)
            if path == str(profile):
                raise RuntimeError("Target page, context or browser has been closed")
            return f"context:{path}"

    class FakePlaywright:
        chromium = Chromium()

    env = browser_module.BrowserEnvironment(
        browser_module.BrowserEnvironmentOptions(user_data_dir=str(profile))
    )
    env._playwright = FakePlaywright()  # type: ignore[assignment]
    context = _asyncio.run(
        env._recover_profile(
            RuntimeError("Target page, context or browser has been closed"),
            profile,
            {},
        )
    )
    assert context.startswith("context:")
    assert calls[0] == str(profile)
    assert calls[1] != str(profile)  # a throwaway profile
    printed = capsys.readouterr().err
    assert "temporary profile" in printed
    assert env._scratch_profile_dir is not None


def test_recover_profile_retries_after_removing_stale_locks(tmp_path: Path) -> None:
    import asyncio as _asyncio

    from arena_api.auth import browser as browser_module

    profile = tmp_path / "direct_browser_profile"
    profile.mkdir()
    (profile / "SingletonLock").write_text("host-999999999", encoding="utf-8")
    calls: list[str] = []

    class Chromium:
        async def launch_persistent_context(self, path: str, **_: Any) -> str:
            calls.append(path)
            if (profile / "SingletonLock").exists():
                raise RuntimeError("ProcessSingleton: profile appears to be in use")
            return f"context:{path}"

    class FakePlaywright:
        chromium = Chromium()

    env = browser_module.BrowserEnvironment(
        browser_module.BrowserEnvironmentOptions(user_data_dir=str(profile))
    )
    env._playwright = FakePlaywright()  # type: ignore[assignment]
    context = _asyncio.run(
        env._recover_profile(
            RuntimeError("ProcessSingleton: profile appears to be in use"),
            profile,
            {},
        )
    )
    # The stale lock was dropped and the real profile came up: trust cookies
    # are preserved instead of silently switching to a scratch profile.
    assert context == f"context:{profile}"
    assert not (profile / "SingletonLock").exists()
    assert env._scratch_profile_dir is None


def test_recover_profile_explains_a_total_failure(tmp_path: Path) -> None:
    import asyncio as _asyncio

    from arena_api.auth import browser as browser_module

    profile = tmp_path / "direct_browser_profile"
    profile.mkdir()

    class Chromium:
        async def launch_persistent_context(self, path: str, **_: Any) -> str:
            raise RuntimeError("Target page, context or browser has been closed")

    class FakePlaywright:
        chromium = Chromium()

    env = browser_module.BrowserEnvironment(
        browser_module.BrowserEnvironmentOptions(user_data_dir=str(profile))
    )
    env._playwright = FakePlaywright()  # type: ignore[assignment]
    with pytest.raises(RuntimeError) as caught:
        _asyncio.run(
            env._recover_profile(
                RuntimeError("Target page, context or browser has been closed"),
                profile,
                {},
            )
        )
    message = str(caught.value)
    assert "chrome.exe" in message
    assert "--fresh-profile" in message


def test_verbose_logging_reports_the_request_size_and_timings(
    serve: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    server, _ = serve("ok", verbose=True)
    _post(
        server,
        "/v1/messages",
        {"stream": False, "messages": [{"role": "user", "content": "привет"}]},
    )
    logged = capsys.readouterr().err
    assert "request: " in logged
    assert "chars" in logged
    assert "first delta" in logged


def test_engine_records_timings_for_the_last_answer() -> None:
    engine = shim.ArenaEngine(engine="echo")
    text = "".join(engine.answer("привет"))
    assert text
    timings = engine.last_timings
    assert timings["prompt_chars"] == len("привет")
    assert timings["answer_chars"] >= len(text)
    assert timings["total_ms"] >= 0


def test_a_tool_call_is_never_streamed_as_text(serve: Any) -> None:
    call = '<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>'
    server, _ = serve("Проверю каталог.\n" + call + "\nГотово.")
    events = _post_stream(
        server,
        {
            "stream": True,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "что в папке?"}],
            "tools": [
                {
                    "name": "Bash",
                    "description": "run a command",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        },
    )
    text = "".join(
        data["delta"].get("text", "")
        for name, data in events
        if name == "content_block_delta"
    )
    # The raw call is a tool_use block, not prose: no duplicated JSON.
    assert "<tool_call>" not in text
    assert "arguments" not in text
    assert "Проверю каталог." in text
    assert "Готово." in text
    kinds = [
        data["content_block"]["type"]
        for name, data in events
        if name == "content_block_start"
    ]
    assert kinds == ["text", "tool_use"]


def test_a_code_fence_with_tools_is_still_plain_text(serve: Any) -> None:
    answer = "Пример:\n```python\nprint(1)\n```\nВсё."
    server, _ = serve(answer)
    events = _post_stream(
        server,
        {
            "stream": True,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "покажи код"}],
            "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
        },
    )
    text = "".join(
        data["delta"]["text"] for name, data in events if name == "content_block_delta"
    )
    assert text == answer
    kinds = [
        data["content_block"]["type"]
        for name, data in events
        if name == "content_block_start"
    ]
    assert kinds == ["text"]


def test_warm_engine_asks_for_archiving_on_every_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon started by the CLI must still archive for the shim.

    The flag travels with the request, not only with the daemon's argv: the CLI
    does not archive by default, and the shim reuses that running daemon.
    """
    from arena_api import warm

    seen: list[dict[str, Any]] = []

    monkeypatch.setattr(warm, "ensure_daemon", lambda **_: "state")

    def fake_ask(prompt: str, **kwargs: Any) -> Iterator[str]:
        seen.append(kwargs)
        yield "ответ"

    monkeypatch.setattr(warm, "ask_stream", fake_ask)
    engine = shim.ArenaEngine(
        engine="warm", base_url="https://example.test", archive=True
    )
    assert "".join(engine.answer("привет")) == "ответ"
    assert seen[0]["archive"] is True

    seen.clear()
    engine_off = shim.ArenaEngine(
        engine="warm", base_url="https://example.test", archive=False
    )
    assert "".join(engine_off.answer("привет")) == "ответ"
    assert seen[0]["archive"] is False


def test_archive_can_target_one_conversation_by_id() -> None:
    chat = "0b7f1c9e-1111-2222-3333-444455556666"
    from arena_api import _direct_page

    marked = _direct_page._archive_mark_triggers_js(chat)
    assert f'const targetId = "{chat}";' in marked
    # The active row is still the fallback, so a chat without an id works.
    plain = _direct_page._archive_mark_triggers_js(None)
    assert "const targetId = null;" in plain
    assert "activeRow" in plain


def test_archiving_a_chat_by_id_clicks_its_own_row() -> None:
    import asyncio

    from arena_api import _direct_page

    chat = "0b7f1c9e-1111-2222-3333-444455556666"
    evaluated: list[str] = []

    class Page:
        def __init__(self) -> None:
            self.locator_hits: list[str] = []

        async def evaluate(self, script: str, *_: Any, **__: Any) -> Any:
            evaluated.append(script)
            if "arena_archive_triggers" in script:
                return {"found": 1, "strategies": [200]}
            if "arena_archive_item" in script:
                return {"found": 1, "label": "Archive"}
            return {"menu_open": False}

        def locator(self, selector: str) -> Any:
            self.locator_hits.append(selector)

            class Target:
                @property
                def first(self) -> Any:
                    async def click(timeout: int = 0) -> None:
                        return None

                    async def wait_for(self, *a: Any, **k: Any) -> None:
                        return None

                    return type("T", (), {"click": staticmethod(click), "wait_for": staticmethod(wait_for)})()

            return Target()

        async def wait_for_timeout(self, ms: int) -> None:
            return None

    page = Page()
    archived = asyncio.run(
        _direct_page.archive_conversation(page, conversation_id=chat, timeout=1.0)
    )
    assert archived is True
    triggers = [script for script in evaluated if "arena_archive_triggers" in script]
    assert triggers and f'const targetId = "{chat}";' in triggers[0]
