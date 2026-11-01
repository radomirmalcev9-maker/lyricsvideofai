"""Apple Motion Lyrics Generator — FastAPI server.

Endpoints:
  POST /api/generate      upload MP3 → Whisper segments → Arena LLM scenes JSON
  GET  /api/scenes/<id>   fetch scenes for a previously uploaded track
  POST /api/export        server-side MP4 rendering (optional, WIP)

Static:  /static/*        (uploads, UI assets)
UI:      /                (static/index.html)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------------------------------- #
# Paths & app bootstrap
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
STATE_DIR = STATIC_DIR / "state"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

# Make the sibling arenaaispofandpars/ importable when running from a checkout.
_REPO_ROOT = BASE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "arenaaispofandpars"))

app = FastAPI(title="Apple Motion Lyrics", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
# Whisper (lazy load — large model)
# --------------------------------------------------------------------------- #

_WHISPER_MODEL = None
_WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")


def get_whisper():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        from faster_whisper import WhisperModel
        _WHISPER_MODEL = WhisperModel(
            _WHISPER_MODEL_SIZE, device="auto", compute_type="auto"
        )
    return _WHISPER_MODEL


# --------------------------------------------------------------------------- #
# Arena client (optional — falls back to stub when not installed/logged in)
# --------------------------------------------------------------------------- #

ARENA_IMPORT_ERROR: str | None = None
try:
    from arena_api import AutoOptions, auto_direct_client  # type: ignore
    _ARENA_OK = True
except Exception as exc:  # noqa: BLE001
    _ARENA_OK = False
    ARENA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def ask_arena(system_prompt: str, user_prompt: str, model: str | None) -> str:
    """Send a single prompt to Arena Direct Chat and return the full text."""
    if not _ARENA_OK:
        return _stub_scenes(user_prompt)

    options = AutoOptions(fast_model=model is None)
    prompt = f"{system_prompt}\n\n{user_prompt}".strip()
    with auto_direct_client(options, verification_token="auto") as client:
        text = client.complete(prompt, model=model) if model else client.complete(prompt)
    return text


# --------------------------------------------------------------------------- #
# Director prompt (Apple/iOS Motion)
# --------------------------------------------------------------------------- #

SYSTEM_DIRECTOR_PROMPT = """\
Ты — арт-директор вертикальных lyric-видео в стиле Apple/iOS Motion (9:16).
Тебе дан JSON со строками песни и их таймингами (start, end, text) после транскрибации Whisper.

Задача: для КАЖДОЙ строки выбрать один iOS-виджет, подходящий по смыслу, настроить его параметры,
выбрать фон и акцентный цвет, и вернуть СТРОГО JSON-массив без markdown (без ```json).

Доступные виджеты (widget) и параметры (params):

1. "face_id"   — экран Face ID. params: {target_state: "scanning"|"success"|"error", progress: 0..100? , subtitle: string? , tags: [string]?}
2. "voice"     — Siri/Распознавание голоса. params: {title: string, subtitle: string, status: "listening"|"found"|"thinking"|"done", accent: string?}
3. "heart_like"— красное сердце на чёрном фоне (Найдено/лайк). params: {title: string, status: string?}
4. "battery"   — батарея iPhone. params: {percent: 0..100, mode: "low"|"normal"|"charging"|"full"}
5. "share_location" — Поделиться геопозицией (два кружка "Я"/"Ты"). params: {me_label: "Я", you_label: "Ты", options: [string], selected: string}
6. "maps"      — карты/навигатор. params: {title: "КАРТЫ", from_point: string?, to_point: string?, distance: string?, subtitle: string?, badge: string?, route_found: bool}
7. "notes"     — Заметки iOS (Записка). params: {title: string, subtitle: string? , body: string?}
8. "photo_note"— Фото·Записка с картинкой. params: {caption: string, quote: string, ps: string?}
9. "palette"   — палитра цветов + кошелёк/валюта. params: {title: string, colors: [{name, hex}], wallet: {label, amount, sub}, accent: string?}
10. "stopwatch"— секундомер. params: {time: string, label: string?}
11. "contacts" — карточка редактирования Контакта iOS. params: {first_name: string, last_name: string?, company: string?, avatar_emoji: string?, avatar_color: string?}
12. "reaction" — выбор смайла-эмоции ("для неё/для него"). params: {title: string, selected: 0|1|2, caption: string?}
13. "imessage"  — групповой чат iMessage. params: {chat_name: string, members: [{short, color}], messages: [{author, text}]}
14. "weather"   — погода/температура. params: {degrees: number, condition: string, city: string?}
15. "player"    — виджет музыкального плеера (крупный). params: {track: string, artist: string, waveform: [n]?}
16. "toggle"    — системный тумблер. params: {label: string, sublabel: string?, on: bool}
17. "timer"     — таймер обратного отсчёта. params: {remaining: string, label: string?}
18. "heart_rate"— пульс/ЭКГ Apple Watch. params: {bpm: number, status: string}

Общие поля сцены:
- scene_id (int, по порядку 1..N)
- start, end (float, копируй из входных таймингов)
- lyric (string, копируй из входной строки)
- widget (один из списка выше)
- params (объект с параметрами виджета)
- theme ("dark"|"light")
- accent (hex-цвет акцента, напр. "#ff453a" — красный, "#30d158" — зелёный, "#0a84ff" — синий, "#bf5af2" — фиолетовый, "#ff9f0a" — оранжевый, "#ff375f" — розовый)
- background: {type: "gradient"|"rain"|"solid"|"noise", colors: [hex, hex]}  — фон под карточку
- highlight_word: string | null  — 1-2 слова из lyric, подсветить цветом accent; можно null

Правила:
- Сохраняй 1:1 порядок строк и тайминги.
- Меняй виджет на каждой смене смысла, но не дроби одну фразу на несколько виджетов — одна строка = одна сцена.
- Подбирай акцент и фон под настроение строки: тёмный градиент для меланхолии, красно-чёрный для агрессии/страсти,
  бело-серый для бытовых сцен (контакты, заметки), дождь+ночь для одиночества.
- Все текстовые значения в params — на русском (названия кнопок, подписи), кроме названий приложений (Face ID, Карты, Заметки).
- Возвращай ТОЛЬКО валидный JSON-массив, без пояснений до и после.
"""


# --------------------------------------------------------------------------- #
# Stub mode — returns a canned scenes array when Arena isn't available.
# Useful for UI development without signing in.
# --------------------------------------------------------------------------- #

_STUB_WIDGETS = ["face_id", "voice", "battery", "notes", "maps", "heart_like",
                 "share_location", "weather", "contacts", "reaction", "player"]
_STUB_ACCENTS = ["#ff453a", "#30d158", "#0a84ff", "#bf5af2", "#ff9f0a", "#ff375f", "#64d2ff"]


def _stub_scenes(user_prompt: str) -> str:
    try:
        match = re.search(r"\[\s*\{.*\}\s*\]", user_prompt, re.DOTALL)
        if not match:
            return "[]"
        subs = json.loads(match.group(0))
    except Exception:
        return "[]"

    import random
    random.seed(42)
    scenes = []
    for i, s in enumerate(subs):
        text = (s.get("text") or "").strip()
        w = _STUB_WIDGETS[i % len(_STUB_WIDGETS)]
        accent = _STUB_ACCENTS[i % len(_STUB_ACCENTS)]
        params: dict[str, Any]
        if w == "battery":
            params = {"percent": int((i / max(1, len(subs) - 1)) * 100), "mode": "normal"}
        elif w == "face_id":
            params = {"target_state": "success", "subtitle": "Распознано"}
        elif w == "voice":
            params = {"title": "Распознавание", "subtitle": "слушаю…", "status": "listening"}
        elif w == "notes":
            params = {"title": "Записка", "subtitle": "суббота, 3:14", "body": text}
        elif w == "maps":
            params = {"title": "КАРТЫ", "distance": "≈ 2 мили", "subtitle": text[:40], "route_found": False}
        elif w == "heart_like":
            params = {"title": "Найдено", "status": "ищу совпадения…"}
        elif w == "share_location":
            params = {"me_label": "Я", "you_label": "Ты", "options": ["На 1 час", "До конца дня", "Бессрочно · навечно"], "selected": "Бессрочно · навечно"}
        elif w == "weather":
            params = {"degrees": -3 + i, "condition": "снег", "city": "Москва"}
        elif w == "contacts":
            params = {"first_name": text.split()[0][:12] if text else "—", "avatar_emoji": "😈", "avatar_color": "#bf5af2"}
        elif w == "reaction":
            params = {"title": "для неё", "selected": 0, "caption": "Смеётся"}
        else:
            params = {"track": "Track", "artist": "Artist"}

        words = text.split()
        hw = words[-1] if words else None
        scenes.append({
            "scene_id": i + 1,
            "start": s.get("start", 0.0),
            "end": s.get("end", 0.0),
            "lyric": text,
            "widget": w,
            "params": params,
            "theme": "dark",
            "accent": accent,
            "background": {"type": "gradient", "colors": ["#0b0c10", "#1c1c2b"]},
            "highlight_word": hw,
        })
    return json.dumps(scenes, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# JSON cleaning
# --------------------------------------------------------------------------- #

def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # drop first line (```json or ```)
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    # Also handle cases where prose surrounds the JSON array: try to locate the array.
    text = text.strip()
    if not text.startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return text


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

def _ext_from_filename(name: str) -> str:
    _, ext = os.path.splitext(name)
    return ext.lower() or ".mp3"


@app.post("/api/generate")
async def generate_lyrics(
    file: UploadFile = File(...),
    model_name: str = Form("fast"),
):
    # 1. Persist upload
    track_id = uuid.uuid4().hex[:12]
    ext = _ext_from_filename(file.filename or "track.mp3")
    safe_name = f"{track_id}{ext}"
    file_path = UPLOAD_DIR / safe_name
    try:
        with file_path.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)
    finally:
        await file.close()

    # 2. Transcribe with Whisper
    t0 = time.monotonic()
    model = get_whisper()
    segments, info = model.transcribe(
        str(file_path),
        language=os.environ.get("WHISPER_LANG", "ru"),
        word_timestamps=False,
        vad_filter=True,
    )
    subtitles = []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        subtitles.append({
            "start": round(float(s.start), 2),
            "end": round(float(s.end), 2),
            "text": text,
        })
    whisper_ms = int((time.monotonic() - t0) * 1000)

    if not subtitles:
        raise HTTPException(status_code=422, detail="Не удалось распознать аудио.")

    # 3. Ask Arena (or stub)
    t1 = time.monotonic()
    user_payload = (
        "Вот строки и тайминги трека:\n"
        + json.dumps(subtitles, ensure_ascii=False, indent=2)
        + "\n\nСоздай сценарий виджетов для каждой строки."
    )
    raw_text = ask_arena(SYSTEM_DIRECTOR_PROMPT, user_payload, model_name or None)
    arena_ms = int((time.monotonic() - t1) * 1000)

    cleaned = _strip_markdown_fence(raw_text)
    try:
        scenes = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=502, content={
            "error": "Модель вернула невалидный JSON.",
            "raw_preview": raw_text[:1000],
            "json_error": str(e),
        })

    if not isinstance(scenes, list):
        raise HTTPException(status_code=502, detail="Ожидался массив сцен.")

    # 4. Save state so it can be retrieved later
    state_path = STATE_DIR / f"{track_id}.json"
    state_path.write_text(
        json.dumps({
            "track_id": track_id,
            "audio": f"/static/uploads/{safe_name}",
            "subtitles": subtitles,
            "scenes": scenes,
            "timings_ms": {"whisper": whisper_ms, "arena": arena_ms},
            "stub_mode": not _ARENA_OK,
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    return JSONResponse({
        "track_id": track_id,
        "audio_url": f"/static/uploads/{safe_name}",
        "scenes": scenes,
        "subtitles": subtitles,
        "timings_ms": {"whisper": whisper_ms, "arena": arena_ms},
        "stub_mode": not _ARENA_OK,
        "arena_status": ARENA_IMPORT_ERROR if not _ARENA_OK else "ok",
    })


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "arena": "ok" if _ARENA_OK else "unavailable",
        "arena_error": ARENA_IMPORT_ERROR,
        "whisper_model": _WHISPER_MODEL_SIZE,
    }


@app.get("/")
def root():
    return JSONResponse({
        "app": "Apple Motion Lyrics",
        "ui": "/static/index.html",
        "health": "/api/health",
    })


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "4770"))
    print(f"\n🎵  Apple Motion Lyrics running at http://{host}:{port}/static/index.html\n")
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=bool(int(os.environ.get("RELOAD", "1"))),
    )
