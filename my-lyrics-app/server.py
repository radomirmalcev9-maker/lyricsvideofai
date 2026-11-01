"""Apple Motion Lyrics Generator — FastAPI server.

Flow:
  1. User uploads MP3 + lyrics text (one visual line per linebreak).
  2. faster-whisper transcribes with word_timestamps=True.
  3. Aligner maps every user-supplied word to the nearest Whisper word,
     giving per-word start/end times per scene-line.
  4. Scenes are sent to the LLM in SMALL BATCHES (a handful of lines at a
     time so each response fits well under the queue timeout). The LLM
     invents an iOS-style widget per scene as self-contained HTML+CSS.
  5. Batches are stitched back together by scene_id and returned to the UI.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import shutil
import uuid
import difflib
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------------------------------- #
# Paths & bootstrap
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
STATE_DIR = STATIC_DIR / "state"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

_REPO_ROOT = BASE_DIR.parent
if str(_REPO_ROOT / "arenaaispofandpars") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "arenaaispofandpars"))

app = FastAPI(title="Apple Motion Lyrics", version="0.3.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
# Whisper (lazy)
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
# Arena client (lazy)
# --------------------------------------------------------------------------- #

ARENA_IMPORT_ERROR: str | None = None
try:
    from arena_api import AutoOptions, auto_direct_client  # type: ignore
    from arena_api.models import GenerationConfig  # type: ignore
    from arena_api.exceptions import QueueTimeoutError  # type: ignore
    _ARENA_OK = True
except Exception as exc:  # noqa: BLE001
    _ARENA_OK = False
    ARENA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    # define fallbacks so the rest of server.py can import without arena installed
    class GenerationConfig:  # type: ignore
        def __init__(self, queue_timeout=180.0, extra=None):
            self.queue_timeout = queue_timeout
            self.extra = extra or {}
    class QueueTimeoutError(Exception):  # type: ignore
        pass


# --------------------------------------------------------------------------- #
# Normalization and aligner
# --------------------------------------------------------------------------- #

_PUNCT_RE = re.compile(r"[^\w0-9а-яА-ЯёЁa-zA-Z]+", flags=re.UNICODE)


def _norm(s: str) -> str:
    return _PUNCT_RE.sub("", (s or "").lower()).strip()


def transcribe_words(audio_path: Path, language: str | None = None) -> list[dict]:
    model = get_whisper()
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )
    words: list[dict] = []
    for seg in segments:
        for w in seg.words or []:
            ww = (w.word or "").strip()
            if not ww:
                continue
            words.append({
                "start": round(float(w.start), 3),
                "end":   round(float(w.end), 3),
                "word":  ww,
                "norm":  _norm(ww),
            })
    return words


def align_lyrics(lyrics_text: str, whisper_words: list[dict]) -> list[dict]:
    lines_raw = [ln.strip() for ln in lyrics_text.splitlines()]
    lines_raw = [ln for ln in lines_raw if ln]

    tokens: list[tuple[int, str]] = []
    for li, line in enumerate(lines_raw):
        for tok in re.split(r"(\s+)", line):
            if not tok or tok.isspace():
                continue
            tokens.append((li, tok))

    scenes: list[dict] = [{"line": ln, "words": []} for ln in lines_raw]
    last_real_t = {"start": 0.0, "end": 0.0}
    w_idx = 0
    n_w = len(whisper_words)

    for li, tok in tokens:
        n = _norm(tok)
        if not n:
            s = last_real_t["end"]
            scenes[li]["words"].append({
                "word": tok, "start": round(s, 3),
                "end": round(s + 0.15, 3), "punct": True,
            })
            continue

        best_j = -1
        best_score = 0.0
        for j in range(w_idx, min(w_idx + 80, n_w)):
            ww = whisper_words[j]
            wn = ww["norm"]
            if not wn:
                continue
            if n == wn:
                best_j = j
                best_score = 1.0
                break
            if n.startswith(wn) or wn.startswith(n):
                score = min(len(n), len(wn)) / max(len(n), len(wn))
                if score > best_score:
                    best_score = score
                    best_j = j
                continue
            ratio = difflib.SequenceMatcher(None, n, wn).ratio()
            if ratio > best_score and ratio >= 0.72:
                best_score = ratio
                best_j = j

        if best_j == -1:
            s = last_real_t["end"]
            scenes[li]["words"].append({
                "word": tok, "start": round(s, 3), "end": round(s + 0.3, 3),
                "aligned": False,
            })
            last_real_t = {"start": s, "end": s + 0.3}
            continue

        w = whisper_words[best_j]
        s, e = float(w["start"]), float(w["end"])
        scenes[li]["words"].append({
            "word": tok, "start": round(s, 3), "end": round(e, 3),
            "aligned": True, "match": best_score,
        })
        last_real_t = {"start": s, "end": e}
        w_idx = best_j + 1

    for sc in scenes:
        if not sc["words"]:
            sc["start"] = 0.0
            sc["end"] = 0.0
            continue
        sc["start"] = sc["words"][0]["start"]
        sc["end"] = round(sc["words"][-1]["end"] + 0.15, 3)
    return scenes


# --------------------------------------------------------------------------- #
# Director prompt (NO templates; AI invents widget HTML from scratch per scene)
# --------------------------------------------------------------------------- #

DIRECTOR_SYSTEM_PROMPT = """\
Ты — ведущий моушн-дизайнер и арт-директор вертикальных lyric-видео в стиле iOS / Apple Motion (9:16).
Я буду присылать тебе порции сцен по несколько строк за раз (JSON-массив).
Каждая сцена: {scene_id, start, end, line}.
Тайминги слов мне НЕ НУЖНЫ в ответе — не включай поле words в свои объекты, чтобы экономить место.

Для КАЖДОЙ сцены верни ОБЪЕКТ с теми же scene_id и ДОПОЛНИТЕЛЬНО с полями:
widget_html, theme, accent, background, entrance.
Выход — РОВНО МАССИВ ТОЙ ЖЕ ДЛИНЫ, в ТОМ ЖЕ ПОРЯДКЕ, scene_id сохраняй.

Требования к widget_html:
- Один self-contained фрагмент, обёртка <div class="w" style="...">...</div>.
- Карточка ~820x520px, padding 40-60px, border-radius ~56px, полупрозрачный
  стеклянный фон (backdrop-filter blur), как виджет iOS.
- Шрифт системный: -apple-system, "SF Pro Display".
- Можно <style>.w ...{...}</style> внутри карточки, анимации только CSS @keyframes (transform/opacity).
- НЕ используй <img>, <svg> со сложной графикой, внешние шрифты. Только
  формы (border/radial-gradient/box-shadow), линии, кружки, emoji, простые иконки.
- НЕ вставляй текст песни в виджет (он выводится отдельно). Короткие подписи/цифры можно.
- Виджет должен узнаваемо выглядеть как кусок интерфейса iPhone и быть СМЫСЛОВО
  СВЯЗАН со строкой (экран звонка/удаления контакта/Apple Pay/будильник/погода/
  карты/iMessage/Face ID/реакция/etc). Не повторяй один и тот же виджет в соседних строках.

Остальные поля:
* theme:      "dark" | "light"
* accent:     hex (#0a84ff, #ff375f, #30d158, #bf5af2, ...)
* background: {"type":"gradient"|"rain"|"sunset"|"light"|"solid", "colors":[hex,hex]}
* entrance:   "rise"|"slide"|"fade"|"pop"

Возвращай ТОЛЬКО JSON-МАССИВ. Никакого markdown, никаких ```json, никакого поясняющего текста до/после.
"""

# Shorter follow-up prompt for subsequent batches (saves tokens).
_DIRECTOR_FOLLOWUP_HINT = """\
Продолжаем песню. Тебе приходит следующая порция строк (продолжение).
Оформи каждую сцену ТЕМ ЖЕ форматом, что и раньше: массив объектов с
scene_id + widget_html/theme/accent/background/entrance. Не повторяй виджеты
из предыдущих строк.
"""


# --------------------------------------------------------------------------- #
# Arena call with batching + retries + generous timeout
# --------------------------------------------------------------------------- #

# Send this many scenes per LLM request. Smaller batches = faster, more reliable,
# less chance of hitting the queue timeout.
_BATCH_SIZE = 6
# How many times to retry a batch that times out before giving up.
_BATCH_RETRIES = 2
# Per-request streaming timeout. Widgets are short HTML snippets but the model
# may take a while to produce several of them; 10 minutes covers long tracks.
_QUEUE_TIMEOUT = 600.0
# Between-batch pause (seconds) to give the browser a chance to cool down.
_BATCH_PAUSE = 1.0


def _call_batch(session, batch_llm: list[dict], is_first: bool, model: str | None) -> list[dict]:
    """Send one batch of scenes (MINIMAL shape) over the provided chat session."""
    if is_first:
        user_prompt = (
            DIRECTOR_SYSTEM_PROMPT
            + "\n\n---\n\nПервая порция сцен — оформи каждую как описано выше:\n"
            + json.dumps(batch_llm, ensure_ascii=False, indent=2)
        )
    else:
        user_prompt = (
            _DIRECTOR_FOLLOWUP_HINT
            + "\n" + json.dumps(batch_llm, ensure_ascii=False, indent=2)
        )
    cfg = GenerationConfig(queue_timeout=_QUEUE_TIMEOUT)
    last_err: Exception | None = None
    for attempt in range(_BATCH_RETRIES + 1):
        try:
            # session.send_message is multi-turn; it keeps the same conversation_id
            text = session.send_message(user_prompt)
            break
        except QueueTimeoutError as e:
            last_err = e
            if attempt < _BATCH_RETRIES:
                time.sleep(2.0 + attempt * 2.0)
                continue
            raise
        except Exception as e:
            last_err = e
            raise
    else:
        raise last_err if last_err else RuntimeError("Arena call failed")

    cleaned = _strip_json(text)
    try:
        out = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}; preview: {text[:600]!r}")
    if not isinstance(out, list) or len(out) != len(batch_llm):
        raise ValueError(
            f"Batch size mismatch: sent {len(batch_llm)}, got "
            f"{len(out) if isinstance(out,list) else type(out).__name__}"
        )
    return out


def ask_arena(all_scenes: list[dict], model: str | None) -> list[dict]:
    if not _ARENA_OK:
        return _stub_scenes(all_scenes)

    # Build the LLM-facing payload WITHOUT the per-word array — it only
    # bloats the prompt and is not needed for widget generation.
    llm_scenes = [
        {"scene_id": s["scene_id"], "start": s["start"], "end": s["end"], "line": s["line"]}
        for s in all_scenes
    ]

    options = AutoOptions(fast_model=model is None)
    out_scenes: list[dict] = []
    with auto_direct_client(options, verification_token="auto") as client:
        # ONE multi-turn chat for all batches so Arena keeps context about
        # widget variety/style and does not start a fresh conversation
        # each time (which is what caused the infinite new-chat loop).
        session = client.chat(model=model, config=GenerationConfig(queue_timeout=_QUEUE_TIMEOUT))
        is_first = True
        for i in range(0, len(all_scenes), _BATCH_SIZE):
            batch_full = all_scenes[i : i + _BATCH_SIZE]
            batch_llm  = llm_scenes[i : i + _BATCH_SIZE]
            batch_out  = _call_batch(session, batch_llm, is_first, model)
            is_first = False
            for s_in, s_out in zip(batch_full, batch_out):
                if not isinstance(s_out, dict) or not s_out.get("widget_html"):
                    s_out = _ensure_widget(s_in, s_out if isinstance(s_out, dict) else {})
                # Force-internal fields (LLM may omit/alter them)
                s_out["scene_id"] = s_in["scene_id"]
                s_out["start"]    = s_in["start"]
                s_out["end"]      = s_in["end"]
                s_out["line"]     = s_in["line"]
                s_out["words"]    = s_in["words"]
                s_out.setdefault("theme", "dark")
                s_out.setdefault("accent", "#0a84ff")
                s_out.setdefault("background", {"type": "gradient", "colors": ["#0b0c10", "#1c1c2b"]})
                s_out.setdefault("entrance", "rise")
                out_scenes.append(s_out)
            if i + _BATCH_SIZE < len(all_scenes):
                time.sleep(_BATCH_PAUSE)
    return out_scenes


def _ensure_widget(s_in: dict, partial: dict) -> dict:
    accent = partial.get("accent", "#0a84ff")
    theme = partial.get("theme", "dark")
    return {
        **partial,
        "theme": theme,
        "accent": accent,
        "background": partial.get("background", {"type": "gradient", "colors": ["#0b0c10", "#1c1c2b"]}),
        "entrance": partial.get("entrance", "rise"),
        "widget_html": (
            f'<div class="w" style="background:rgba(28,28,30,.72);border:1.5px solid rgba(255,255,255,.12);'
            f'border-radius:56px;padding:48px;color:#fff;-webkit-backdrop-filter:blur(50px) saturate(140%);'
            f'backdrop-filter:blur(50px) saturate(140%);box-shadow:0 40px 100px rgba(0,0,0,.55);">'
            f'<div style="font-size:28px;color:rgba(255,255,255,.6);font-weight:700;">SCENE {s_in["scene_id"]}</div>'
            f'<div style="margin-top:40px;font-size:64px;font-weight:800;color:{accent};">AI</div>'
            f'<div style="margin-top:16px;font-size:26px;color:rgba(255,255,255,.55);">placeholder widget</div>'
            f'</div>'
        ),
    }


def _stub_scenes(scenes: list[dict]) -> list[dict]:
    """Offline stub: deterministic placeholder widgets so the UI works without Arena."""
    import random
    random.seed(7)
    accents = ["#0a84ff", "#ff375f", "#30d158", "#bf5af2", "#ff9f0a", "#64d2ff", "#ffd60a"]
    out = []
    for i, sc in enumerate(scenes):
        accent = accents[i % len(accents)]
        out.append(_ensure_widget(sc, {
            "theme": "dark",
            "accent": accent,
            "background": {"type": "gradient", "colors": ["#0b0c10", "#1c1c2b"]},
            "entrance": "rise",
        }))
    return out


def _strip_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()
    if not text.startswith("["):
        s = text.find("[")
        e = text.rfind("]")
        if s != -1 and e != -1 and e > s:
            text = text[s : e + 1]
    return text


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

def _safe_name(name: str) -> str:
    base = os.path.basename(name or "track.mp3")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base) or "track.mp3"


@app.post("/api/generate")
async def generate_lyrics(
    file: UploadFile = File(...),
    lyrics: str = Form(...),
    model_name: str = Form("fast"),
    language: str = Form("ru"),
):
    lyrics = (lyrics or "").strip()
    if not lyrics:
        raise HTTPException(status_code=422, detail="Нужен текст песни (поле lyrics).")

    # 1. Save audio
    track_id = uuid.uuid4().hex[:12]
    ext = os.path.splitext(file.filename or "track.mp3")[1].lower() or ".mp3"
    safe_name = f"{track_id}{ext}"
    file_path = UPLOAD_DIR / safe_name
    try:
        with file_path.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)
    finally:
        await file.close()

    # 2. Whisper word timings
    t0 = time.monotonic()
    try:
        whisper_words = transcribe_words(file_path, language or None)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Whisper error: {exc}")
    whisper_ms = int((time.monotonic() - t0) * 1000)

    # 3. Align user lyrics → scenes with per-word timings
    scenes = align_lyrics(lyrics, whisper_words)
    if not scenes:
        raise HTTPException(status_code=422, detail="Не удалось разбить текст на строки.")

    # Build numbered scenes (with words) to pass to ask_arena.
    # ask_arena itself strips `words` before sending to the LLM to save
    # tokens and avoid hitting input/context limits.
    numbered_scenes = [
        {
            "scene_id": i + 1,
            "start":    s["start"],
            "end":      s["end"],
            "line":     s["line"],
            "words":    s["words"],
        }
        for i, s in enumerate(scenes)
    ]

    # 4. Ask Arena for widget_html (batched, single chat session)
    t1 = time.monotonic()
    arena_err: str | None = None
    try:
        scenes_out = ask_arena(numbered_scenes, model_name or None)
    except QueueTimeoutError as e:
        arena_err = (
            f"Превышен таймаут Arena ({_QUEUE_TIMEOUT:.0f}s). Попробуй быструю модель "
            f"или разбей трек на части. ({e})"
        )
        scenes_out = _stub_scenes(numbered_scenes)
    except Exception as exc:  # noqa: BLE001
        arena_err = f"Ошибка Arena: {exc}"
        scenes_out = _stub_scenes(numbered_scenes)
    arena_ms = int((time.monotonic() - t1) * 1000)

    state_path = STATE_DIR / f"{track_id}.json"
    state_path.write_text(json.dumps({
        "track_id": track_id,
        "audio": f"/static/uploads/{safe_name}",
        "scenes": scenes_out,
        "timings_ms": {"whisper": whisper_ms, "arena": arena_ms},
        "stub_mode": (not _ARENA_OK) or (arena_err is not None),
        "arena_error": arena_err,
    }, ensure_ascii=False), encoding="utf-8")

    return JSONResponse({
        "track_id": track_id,
        "audio_url": f"/static/uploads/{safe_name}",
        "scenes": scenes_out,
        "timings_ms": {"whisper": whisper_ms, "arena": arena_ms},
        "stub_mode": (not _ARENA_OK) or (arena_err is not None),
        "arena_status": ARENA_IMPORT_ERROR if not _ARENA_OK else (arena_err or "ok"),
    })


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "arena": "ok" if _ARENA_OK else "unavailable",
        "arena_error": ARENA_IMPORT_ERROR,
        "whisper_model": _WHISPER_MODEL_SIZE,
        "queue_timeout_s": _QUEUE_TIMEOUT,
        "batch_size": _BATCH_SIZE,
    }


@app.get("/", response_class=HTMLResponse)
def root():
    return (
        "<!doctype html><meta charset=utf-8>"
        "<title>Apple Motion Lyrics</title>"
        '<meta http-equiv="refresh" content="0; url=/static/index.html">'
        "<p>Redirecting to <a href=/static/index.html>/static/index.html</a>..."
    )


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "4770"))
    print()
    print("  [Apple Motion Lyrics]")
    print(f"  Open: http://{host}:{port}/static/index.html")
    print()
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=bool(int(os.environ.get("RELOAD", "1"))),
    )
