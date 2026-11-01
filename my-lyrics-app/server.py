"""Apple Motion Lyrics Generator — FastAPI server.

Flow:
  1. User uploads MP3 + lyrics text (one visual line per linebreak).
  2. faster-whisper transcribes with word_timestamps=True.
  3. A lightweight aligner maps every word of the user-supplied lyrics
     to the nearest Whisper word by normalized-text substring/fuzzy match,
     giving per-word start/end times per scene-line.
  4. Each line (with per-word timings) goes to the LLM as a JSON scene.
  5. The LLM invents an iOS-style widget from scratch as a self-contained
     HTML+CSS snippet (no pre-defined templates).
  6. Frontend plays audio, highlights words in karaoke fashion and injects
     each scene's widget HTML into the glass card at the right time.
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
# Paths & app bootstrap
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
STATE_DIR = STATIC_DIR / "state"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

# Make the sibling arenaaispofandpars/ importable.
_REPO_ROOT = BASE_DIR.parent
if str(_REPO_ROOT / "arenaaispofandpars") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "arenaaispofandpars"))

app = FastAPI(title="Apple Motion Lyrics", version="0.2.0")
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
# Arena client
# --------------------------------------------------------------------------- #

ARENA_IMPORT_ERROR: str | None = None
try:
    from arena_api import AutoOptions, auto_direct_client  # type: ignore
    _ARENA_OK = True
except Exception as exc:  # noqa: BLE001
    _ARENA_OK = False
    ARENA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Text normalization + aligner
# --------------------------------------------------------------------------- #

_PUNCT_RE = re.compile(r"[^\w0-9а-яА-ЯёЁa-zA-Z]+", flags=re.UNICODE)


def _norm(s: str) -> str:
    """Normalize a word for loose matching: lowercase, strip punctuation."""
    return _PUNCT_RE.sub("", (s or "").lower()).strip()


def transcribe_words(audio_path: Path, language: str | None = None) -> list[dict]:
    """Return a list of Whisper word dicts: {start, end, word, norm}."""
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
    """Split lyrics into visual lines and assign timings to every word.

    Strategy: walk the user lyrics word-by-word. For each word look through
    Whisper words that haven't been consumed yet and pick the best match
    (first by exact normalized equality, falling back to difflib ratio
    >0.72; punctuation-only "words" inherit the timing of the previous
    real word). Lines split on '\\n' become scenes.
    """
    lines_raw = [ln.strip() for ln in lyrics_text.splitlines()]
    lines_raw = [ln for ln in lines_raw if ln]

    # Tokenize lyrics into (line_index, original_token)
    tokens: list[tuple[int, str]] = []
    for li, line in enumerate(lines_raw):
        # Split into whitespace-separated tokens but keep punctuation attached.
        for tok in re.split(r"(\s+)", line):
            if not tok or tok.isspace():
                continue
            tokens.append((li, tok))

    scenes: list[dict] = [{"line": ln, "words": []} for ln in lines_raw]

    # Punctuation-only tokens (commas, dashes, ellipsis) get their timing
    # from the previous real word.
    last_real_t = {"start": 0.0, "end": 0.0}
    w_idx = 0  # cursor in whisper_words
    n_w = len(whisper_words)

    for li, tok in tokens:
        n = _norm(tok)
        if not n:
            # punctuation-only token: attach to previous timing, duration ~0
            s = last_real_t["end"]
            scenes[li]["words"].append({
                "word": tok, "start": round(s, 3),
                "end": round(s + 0.15, 3), "punct": True,
            })
            continue

        best_j = -1
        best_score = 0.0
        # Search in a rolling window of upcoming Whisper words
        for j in range(w_idx, min(w_idx + 60, n_w)):
            ww = whisper_words[j]
            wn = ww["norm"]
            if not wn:
                continue
            if n == wn:
                best_j = j
                best_score = 1.0
                break
            # prefix/substring
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
            # Fall back to projecting linearly from the last known timing.
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

    # Scene-level start/end = first/last word times
    for sc in scenes:
        if not sc["words"]:
            sc["start"] = 0.0
            sc["end"] = 0.0
            continue
        sc["start"] = sc["words"][0]["start"]
        sc["end"] = sc["words"][-1]["end"]
        # Small tail so the line doesn't cut off too abruptly.
        sc["end"] = round(sc["end"] + 0.15, 3)
    return scenes


# --------------------------------------------------------------------------- #
# Director prompt — NO widget templates. LLM invents HTML/CSS from scratch.
# --------------------------------------------------------------------------- #

SYSTEM_DIRECTOR_PROMPT = """\
Ты — ведущий моушн-дизайнер, арт-директор вертикальных lyric-видео в стиле iOS / Apple Motion (9:16).

Тебе дан JSON-массив сцен песни. Каждая сцена = одна строка трека, которая показывается на экране
от start до end секунд. У сцены есть words[] с точными таймингами каждого СЛОВА (start/end в секундах).

Твоя задача — для КАЖДОЙ сцены:
1. Выбрать атмосферу (тёмная / светлая), акцентный hex-цвет и абстрактное описание фона
   (ночной дождь, закат, неон-город, студия, туманное небо, белый лист и т.д.).
2. Придумать и СВЕРСТАТЬ виджет с НУЛЯ как self-contained HTML+CSS-врезку — это должен быть
   узнаваемый фрагмент интерфейса iPhone (но не обязательно существующий экран!), максимально
   подходящий по смыслу строки. Можно использовать ЛЮБЫЕ идеи: анимированная иконка, диалог iMessage,
   экран будильника, карточка контакта, фейковый баннер уведомления, платёж, камера, галерея фото,
   фитнес-кольца, экран звонка, погода, таймер, пароль, Face ID, локатор, заметка, голосовой мемо,
   кошелёк, карта, музыкальный плеер, скриншот чата, шкала громкости, авиарежим, компас, календарь,
   напоминание, покупка в App Store — что угодно, ГЛАВНОЕ: виджет должен быть СВЯЗАН СО СМЫСЛОМ СТРОКИ,
   а не случайным шаблоном.
3. Вернуть СТРОГО JSON-массив без markdown и без комментариев, одной JSON-структурой.

Технические требования к HTML виджета (widget_html):
- Обёртка виджета — <div class="w" style="...">. Карточка сама отвечает за свою внутреннюю вёрстку.
- Пространство карточки: ~820×520 px с внутренним padding 40-60 px, скругление углов ~56px,
  backdrop-filter blur, полупрозрачный тёмный или светлый фон под стать theme.
- Шрифты системные: -apple-system, SF Pro Display. Размеры шрифтов в px (под 1080×1920).
- Можно использовать inline <style> только внутри виджета, ограничивая правила .w .что-то.
- В анимациях используй только CSS @keyframes (transform / opacity).
- НЕ ИСПОЛЬЗУЙ картинок, <img>, <svg> со сложной графикой, шрифтов из интернета — только формы,
  линии, градиенты, кружки, emoji, простые геометрические иконки через border/radial-gradient/box-shadow.
- НЕ вставляй сам текст песни в виджет — текст песни рендерится отдельно под/над виджетом.
  Виджет может содержать короткие свои подписи (название экрана, цифры, имена кнопок).

Поля каждой сцены (копируй start/end/words из входа как есть):
- scene_id (int, начиная с 1)
- start, end           — копируй из входа
- line                — копируй текст строки
- words               — массив словоформ как во входе, не изменяй тайминги
- widget_html         — string, один self-contained HTML-фрагмент карточки с inline-стилями
- theme               — "dark" | "light"
- accent              — hex-цвет, используется для подсветки текста и акцентов интерфейса
- background          — {type: "gradient"|"rain"|"solid"|"sunset", colors: [hex, hex]}
- entrance            — "rise" | "slide" | "fade" | "pop" — как карточка появляется на строке

Возвращай ВАЛИДНЫЙ JSON. Без текста до и после. Без ```json-обёртки.
"""


# --------------------------------------------------------------------------- #
# Arena call
# --------------------------------------------------------------------------- #

def ask_arena(system_prompt: str, user_payload: list[dict], model: str | None) -> str:
    if not _ARENA_OK:
        return _stub_scenes(user_payload)
    options = AutoOptions(fast_model=model is None)
    user_prompt = (
        "Вот сцены с таймингами слов:\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
        + "\n\nСгенерируй widget_html для КАЖДОЙ сцены как описано в инструкции."
    )
    prompt = f"{system_prompt}\n\n{user_prompt}"
    with auto_direct_client(options, verification_token="auto") as client:
        if model:
            text = client.complete(prompt, model=model)
        else:
            text = client.complete(prompt)
    return text


# --------------------------------------------------------------------------- #
# Stub (for development without Arena) — minimal placeholder widgets,
# still real HTML so the frontend can render scenes even offline.
# --------------------------------------------------------------------------- #

def _stub_scenes(scenes: list[dict]) -> str:
    import random
    random.seed(7)
    accents = ["#0a84ff", "#ff375f", "#30d158", "#bf5af2", "#ff9f0a", "#64d2ff", "#ffd60a"]
    out = []
    for i, sc in enumerate(scenes):
        accent = accents[i % len(accents)]
        theme = "dark"
        bg_colors = ["#0b0c10", "#1c1c2b"]
        html = (
            f'<div class="w" style="background:rgba(28,28,30,.72);border:1.5px solid rgba(255,255,255,.12);'
            f'border-radius:56px;padding:48px;color:#fff;-webkit-backdrop-filter:blur(50px) saturate(140%);'
            f'backdrop-filter:blur(50px) saturate(140%);box-shadow:0 40px 100px rgba(0,0,0,.55);">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div style="font-size:28px;color:rgba(255,255,255,.6);font-weight:700;letter-spacing:.04em;">SCENE {i+1}</div>'
            f'<div style="font-size:22px;color:{accent};font-weight:700;">LIVE</div>'
            f'</div>'
            f'<div style="margin-top:40px;font-size:76px;font-weight:800;letter-spacing:-.03em;">{accent}</div>'
            f'<div style="margin-top:20px;height:18px;background:rgba(255,255,255,.12);border-radius:12px;overflow:hidden;">'
            f'<div style="width:{int((i+1)/max(1,len(scenes))*100)}%;height:100%;background:{accent};"></div></div>'
            f'<div style="margin-top:30px;font-size:26px;color:rgba(255,255,255,.55);">Stub-виджет — подключи Arena для настоящих.</div>'
            f'</div>'
        )
        out.append({
            "scene_id": i + 1,
            "start": sc["start"],
            "end": sc["end"],
            "line": sc["line"],
            "words": sc["words"],
            "widget_html": html,
            "theme": theme,
            "accent": accent,
            "background": {"type": "gradient", "colors": bg_colors},
            "entrance": "rise",
        })
    return json.dumps(out, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# JSON cleanup
# --------------------------------------------------------------------------- #

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
    if not lyrics or not lyrics.strip():
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

    # 2. Whisper word timestamps
    t0 = time.monotonic()
    try:
        whisper_words = transcribe_words(file_path, language or None)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Whisper failed: {exc}")
    whisper_ms = int((time.monotonic() - t0) * 1000)

    # 3. Align user lyrics to Whisper words (per-word timings per line)
    scenes = align_lyrics(lyrics, whisper_words)
    if not scenes:
        raise HTTPException(status_code=422, detail="Не удалось разбить текст на строки.")

    # Build the payload we send to the LLM (only structural fields, no html yet).
    payload = [
        {
            "scene_id": i + 1,
            "start": s["start"],
            "end": s["end"],
            "line": s["line"],
            "words": s["words"],
        }
        for i, s in enumerate(scenes)
    ]

    # 4. Ask Arena to invent widget_html for each scene
    t1 = time.monotonic()
    raw_text = ask_arena(SYSTEM_DIRECTOR_PROMPT, payload, model_name or None)
    arena_ms = int((time.monotonic() - t1) * 1000)

    cleaned = _strip_json(raw_text)
    try:
        scenes_out = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=502, content={
            "error": "Модель вернула невалидный JSON.",
            "raw_preview": raw_text[:1500],
            "json_error": str(e),
        })

    if not isinstance(scenes_out, list):
        raise HTTPException(status_code=502, detail="Ожидался массив сцен.")

    # 5. Persist state
    state_path = STATE_DIR / f"{track_id}.json"
    state_path.write_text(json.dumps({
        "track_id": track_id,
        "audio": f"/static/uploads/{safe_name}",
        "scenes": scenes_out,
        "timings_ms": {"whisper": whisper_ms, "arena": arena_ms},
        "stub_mode": not _ARENA_OK,
    }, ensure_ascii=False), encoding="utf-8")

    return JSONResponse({
        "track_id": track_id,
        "audio_url": f"/static/uploads/{safe_name}",
        "scenes": scenes_out,
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
