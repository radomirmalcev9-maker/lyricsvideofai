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
    """Aggressive normalization for fuzzy matching (case, punctuation, ё→е, ъ/ь)."""
    t = _PUNCT_RE.sub("", (s or "").lower())
    t = t.replace("ё", "е")
    # Drop soft/hard signs for Russian (Whisper often omits them).
    t = t.replace("ъ", "").replace("ь", "")
    return t.strip()


def _word_similarity(a: str, b: str) -> float:
    """Return 0..1 similarity between two NORMALIZED words."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # One is prefix/suffix of the other (e.g. Whisper glued two words).
    if len(a) >= 3 and len(b) >= 3:
        if a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a):
            return min(len(a), len(b)) / max(len(a), len(b)) * 0.92
    return difflib.SequenceMatcher(None, a, b).ratio()


def transcribe_words(audio_path: Path, language: str | None = None) -> list[dict]:
    model = get_whisper()
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        # Don't let VAD chop off word tails at line boundaries.
        vad_filter=False,
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
    """Globally align user-supplied lyrics lines to Whisper word timings.

    Algorithm:
      1. Tokenize the lyrics into (line_index, raw_token), keeping punctuation
         tokens attached to their line but flagging them.
      2. Build a sequence of "real" (non-punct) lyrics tokens and a matching
         sequence of Whisper words (already normalized).
      3. Run a Needleman-Wunsch-style global DP alignment between the two
         sequences (match/mismatch cost by word similarity; linear gap
         penalties for insertions/deletions on either side). This guarantees
         the alignment is globally optimal AND monotonic — no more "one
         missed word shifts everything after it" bug.
      4. Backtrace to find the 1-to-1 pairs (lyrics_token_idx ↔ whisper_idx).
      5. Interpolate timings for tokens with no matched whisper word
         (punctuation, missed words, repeated articles) linearly between
         their nearest matched neighbors.
      6. Group tokens back into lines to produce scenes.
    """
    # --- 1. Tokenize lyrics preserving line breaks ---
    lines_raw = [ln.strip() for ln in lyrics_text.splitlines()]
    lines_raw = [ln for ln in lines_raw if ln]

    TokRec = tuple  # (line_idx, raw_text, norm_text, is_punct)
    tokens: list[tuple[int, str, str, bool]] = []
    for li, line in enumerate(lines_raw):
        for tok in re.split(r"(\s+)", line):
            if not tok or tok.isspace():
                continue
            n = _norm(tok)
            tokens.append((li, tok, n, not n))  # punct-only token if normalized is empty

    # Real-word tokens only (punctuation has no Whisper counterpart).
    real_positions = [i for i, t in enumerate(tokens) if not t[3]]
    if not whisper_words or not real_positions:
        # Fallback: zero-length scenes.
        return [{"line": ln, "words": [], "start": 0.0, "end": 0.0} for ln in lines_raw]

    # --- 2. Prepare sequences ---
    N = len(real_positions)          # lyrics real words
    M = len(whisper_words)           # whisper words
    lyrics_norms = [tokens[k][2] for k in real_positions]
    whisper_norms = [w["norm"] for w in whisper_words]

    # DP score matrix. We maximize similarity, with a fixed penalty per gap.
    # GAP_OPEN penalizes skipping a word on either side. Lower it so the
    # aligner is willing to skip filler/hesitation words Whisper invents.
    GAP_PEN = -0.45
    MATCH_MULT = 1.0  # weight of similarity score for a pair
    # dp[i][j] = best score aligning first i lyrics words to first j whisper words
    dp: list[list[float]] = [[0.0] * (M + 1) for _ in range(N + 1)]
    # back: 'D' = diagonal (match), 'U' = up (skip whisper, j-1), 'L' = left (skip lyrics, i-1)
    bt: list[list[str]] = [[""] * (M + 1) for _ in range(N + 1)]
    NEG_INF = float("-inf")
    for i in range(1, N + 1):
        dp[i][0] = dp[i - 1][0] + GAP_PEN
        bt[i][0] = "L"
    for j in range(1, M + 1):
        dp[0][j] = dp[0][j - 1] + GAP_PEN * 0.6  # slightly cheaper to skip whisper fillers
        bt[0][j] = "U"

    # Banded DP: only consider |i/M - j/N| < BAND relative to diagonal to keep
    # things O((N+M)*BAND). Russian lyrics have ~same number of words as Whisper
    # sees, so a generous band of ±80 covers bridges/hesitations safely.
    BAND = 120

    for i in range(1, N + 1):
        j_lo = max(1, int(i * M / N) - BAND)
        j_hi = min(M, int(i * M / N) + BAND)
        for j in range(j_lo, j_hi + 1):
            sim = _word_similarity(lyrics_norms[i - 1], whisper_norms[j - 1])
            match_score = dp[i - 1][j - 1] + MATCH_MULT * sim
            # Skip a lyrics word (deletion in Whisper)
            skipL = dp[i - 1][j] + GAP_PEN if bt[i - 1][j] else NEG_INF
            # Skip a whisper word (insertion/filler)
            skipU = dp[i][j - 1] + GAP_PEN * 0.6 if bt[i][j - 1] else NEG_INF
            best = max(match_score, skipL, skipU)
            dp[i][j] = best
            if best == match_score:
                bt[i][j] = "D"
            elif best == skipL:
                bt[i][j] = "L"
            else:
                bt[i][j] = "U"
        # Make sure all columns have reachable values (fill out of band with NEG_INF)
        for j in range(0, M + 1):
            if not bt[i][j] and j != 0:
                dp[i][j] = NEG_INF

    # --- 4. Backtrace from (N, M) ---
    pairs: dict[int, int] = {}  # lyrics_real_index -> whisper_index
    i, j = N, M
    # If endpoint is unreachable (extremely rare), find the best reachable cell on row N or col M.
    if dp[i][j] == NEG_INF:
        best_v = NEG_INF
        best_ij = (N, M)
        for jj in range(0, M + 1):
            if dp[N][jj] > best_v:
                best_v = dp[N][jj]; best_ij = (N, jj)
        for ii in range(0, N + 1):
            if dp[ii][M] > best_v:
                best_v = dp[ii][M]; best_ij = (ii, M)
        i, j = best_ij
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] == "D":
            pairs[i - 1] = j - 1
            i -= 1; j -= 1
        elif i > 0 and bt[i][j] == "L":
            i -= 1  # lyrics word not found in whisper
        elif j > 0 and bt[i][j] == "U":
            j -= 1  # whisper word unused
        else:
            break

    # --- 5. Assign times to every token (real words + punctuation) ---
    # Build a list of (token_idx_absolute, start, end) for real aligned words.
    anchor: list[tuple[int, float, float]] = []
    for ri, tk_idx in enumerate(real_positions):
        wi = pairs.get(ri)
        if wi is None:
            continue
        w = whisper_words[wi]
        anchor.append((tk_idx, float(w["start"]), float(w["end"])))

    # Sort anchors by time (should already be monotonic by DP construction).
    anchor.sort(key=lambda x: x[1])

    # Fallback anchor if nothing aligned (degenerate).
    if not anchor:
        t0 = 0.0
        for tk_idx, (li, raw, n, is_punct) in enumerate(tokens):
            anchor.append((tk_idx, t0, t0 + 0.25))
            t0 += 0.25
    else:
        # Extrapolate before first anchor and after last anchor at constant pace.
        first_idx, first_s, first_e = anchor[0]
        last_idx,  last_s,  last_e  = anchor[-1]
        # Pace (seconds per token) estimated from anchors.
        if len(anchor) >= 2:
            span_t = last_e - first_s
            span_n = anchor[-1][0] - first_idx + 1 or 1
            pace = span_t / span_n
        else:
            pace = 0.28
        # Pad before
        pad = []
        t = first_s
        for k in range(first_idx - 1, -1, -1):
            t -= pace
            pad.append((k, max(0.0, t), max(0.0, t + pace)))
        pad.reverse()
        anchor = pad + anchor
        # Pad after
        t = last_e
        pad_after = []
        for k in range(last_idx + 1, len(tokens)):
            pad_after.append((k, t, t + pace))
            t += pace
        anchor = anchor + pad_after

    # Build a map token_idx -> (start, end) via linear interpolation between anchors.
    times: dict[int, tuple[float, float]] = {}
    for a_pos in range(len(anchor) - 1):
        i0, s0, e0 = anchor[a_pos]
        i1, s1, e1 = anchor[a_pos + 1]
        times[i0] = (s0, e0)
        gap = i1 - i0
        if gap > 1:
            for d in range(1, gap):
                frac = d / gap
                st = s0 + (s1 - s0) * frac
                en = e0 + (e1 - e0) * frac
                # Guard against zero/negative durations
                if en <= st:
                    en = st + 0.12
                times[i0 + d] = (st, en)
    times[anchor[-1][0]] = (anchor[-1][1], anchor[-1][2])

    # Map token_idx -> real_index (for non-punct tokens)
    tok_to_real = {tk_idx: ri for ri, tk_idx in enumerate(real_positions)}

    # --- 6. Group back into scenes/lines ---
    scenes: list[dict] = [{"line": ln, "words": []} for ln in lines_raw]
    for tk_idx, (li, raw, n, is_punct) in enumerate(tokens):
        s, e = times.get(tk_idx, (0.0, 0.15))
        rec = {
            "word": raw,
            "start": round(s, 3),
            "end":   round(max(e, s + 0.05), 3),
        }
        if is_punct:
            rec["punct"] = True
        else:
            ri = tok_to_real.get(tk_idx, -1)
            rec["aligned"] = ri in pairs
            if ri in pairs:
                wn = whisper_norms[pairs[ri]]
                rec["match"] = round(_word_similarity(n, wn), 3)
        scenes[li]["words"].append(rec)

    for sc in scenes:
        if not sc["words"]:
            sc["start"] = 0.0
            sc["end"]   = 0.0
            continue
        sc["start"] = sc["words"][0]["start"]
        sc["end"]   = round(sc["words"][-1]["end"] + 0.15, 3)

    return scenes


# --------------------------------------------------------------------------- #
# Director prompt (NO templates; AI invents widget HTML from scratch per scene)
# --------------------------------------------------------------------------- #

DIRECTOR_SYSTEM_PROMPT = """\
Ты — Арт-директор Apple, ведущий дизайнер интерфейсов iOS 26 (liquid-glass) и моушн-дизайнер.
Рисуешь НЕ абстрактные красивые квадраты, а УЗНАВАЕМЫЕ ФРАГМЕНТЫ РЕАЛЬНОГО ИНТЕРФЕЙСА iPhone —
такие, чтобы пользователь глядя на кадр сразу понял: «о, это экран звонка», «о, это Apple Pay»,
«о, это локскрин с уведомлением», «о, это виджет погоды», «о, это экран Now Playing».

Я присылаю порциями по несколько строк песни (JSON-массив). Каждая сцена = одна строка:
{scene_id, start, end, line}.
Тайминги слов в ответе НЕ НУЖНЫ — не включай поле words.

Для КАЖДОЙ сцены верни ОБЪЕКТ с тем же scene_id + поля widget_html, theme, accent, background, entrance.
Ровно массив той же длины, тот же порядок.

═══════════════════════════════════════════════════════════════
КАЧЕСТВО ВИДЖЕТА — ЭТО ГЛАВНОЕ. ТРЕБОВАНИЯ К РЕАЛИЗАЦИИ:
═══════════════════════════════════════════════════════════════
1. Обёртка: <div class="w"> с ровно ТРЕМЯ вложенными смысловыми слоями:
   • стеклянная карточка-корпус (position:relative; width:820px; min-height:520px;
     border-radius:56px; padding:44px; display:flex; flex-direction:column; gap:18px;
     background: для темы dark — linear-gradient(145deg, rgba(255,255,255,.14), rgba(255,255,255,.04));
     для темы light — linear-gradient(145deg, rgba(255,255,255,.78), rgba(255,255,255,.45));
     border: 1px solid rgba(255,255,255,.22) (dark) / rgba(255,255,255,.65) (light);
     box-shadow: 0 30px 80px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.35);
     backdrop-filter: blur(40px) saturate(180%);
     -webkit-backdrop-filter: blur(40px) saturate(180%);
     overflow:hidden;
   • верхний ярлычок / статус: крупный emoji 56px или кружок-аватар 80px с градиентом,
     рядом подпись: крупный заголовок 40px 700 и под ним мелкая подпись 24px 500
     в акцентном/вторичном цвете.
   • центральный контент-блок: САМАЯ ВИЗУАЛЬНО БОГАТАЯ ЧАСТЬ, 60-70% карточки.
     Не один прямоугольник с текстом, а МНОЖЕСТВО ДЕТАЛЕЙ:
     - кружки с градиентами и radial-highlight (стеклянные «шайбы»);
     - прогресс-кольца (svg-like круги, реализованные через border + conic-gradient);
     - тонкие сепараторы (1px rgba(255,255,255,.12));
     - иконки через комбинацию маленьких квадратов/кругов/линий (кнопка play = ▶,
       пауза = две вертикальные полоски, телефонная трубка — белый круг с зелёным фоном
       и символом, сообщение — голубой «пузырь» с текстом, сердечко — скруглённый
       розовый квадрат повёрнутый на 45deg с псевдо-элементами, и т.д.);
     - «стеклянные» сегменты/табы с 2-3 кнопками;
     - слайдеры/прогресс-бары с «блёсткой» головкой (white radial-gradient).
   • нижняя строка: обычно подпись/метаданные + 1-2 маленьких акцентных кружка-кнопки
     (play/skip/принять/отклонить/селфи/камера).

2. СЕМАНТИЧЕСКАЯ ПРИВЯЗКА. Выбери виджет ПО СМЫСЛУ СТРОКИ, а не абстрактную подложку.
   Вот КАТАЛОГ ПРИМЕРОВ (можно и другие придумать, главное УЗНАВАЕМОСТЬ):
   • любовь/сердце/чувства → экран Face ID / локскрин с сердечком-реакцией / сообщение с сердечком bubble
   • расставание/звонок/не берёт трубку → экран ВХОДЯЩИЙ ЗВОНОК: красная кнопка «отклонить» (круг),
     зелёная «ответить», имя контакта большим шрифтом, мелко «мобильный»
   • деньги/оплата/кошелёк → Apple Pay карта (градиентная карточка с чипом и надписью,
     кружок Face ID, сумма крупно, галочка ✓ в круге при подтверждении)
   • время/ночь/часы/опоздание → виджет ЧАСЫ или БУДИЛЬНИК: циферблат с стрелками
     (через 2 линии, повёрнутые transform:rotate), время крупно, тумблер активный
   • погода/дождь/снег/облака → погодный виджет: температура огромно, город,
     эмоджи/градиент неба, прогноз на 3 точки с температурами
   • дорога/ехать/машина/город → Карты: карточка маршрута с синей линией,
     ETA, «12 мин» крупно, стрелка-поворот в кружке
   • сообщение/смс/написал → iMessage bubble: большой синий (акцент) пузырь
     с белым текстом на 1-3 слова, аватарка в кружке сверху, время
   • фото/воспоминания/вместе → экран Фото: 2-3 квадратика-«фото» с разными
     градиентами и рамками, заголовок «Воспоминания», подпись
   • музыка/песня/слушать → Now Playing: обложка (градиентный квадрат с бликом),
     прогресс-линия с круглой головкой, ▶ / ⏮ / ⏭ кнопки, название и исполнитель
   • камера/смотрю/фотография → Камера: большой круг видоискателя, переключатели режимов,
     белая круглая кнопка спуска снизу
   • уведомление/всплыло → Локскрин-уведомление: время сверху, стеклянный пласт
     с иконкой приложения, 2-строчным текстом, синей полоской внизу
   • ночь/спать/закат/утро → виджет с большой луной/солнцем (радиал-градиент),
     временем, подписью
   • пароль/секрет/разблокировать → экран блокировки: 6 точек кода, Face ID иконка
   • дождь/слёзы/мокрый → виджет погоды с дождём (несколько вертикальных линий-слёз)
   • звезда/небо/сиять → тёмная карточка с точками-звёздами (маленькие белые кружки),
     полумесяц, градиент ночного неба
   • бежать/двигаться/ехать → виджет Фитнес/Активность: три концентрических кольца
     (красное/зелёное/голубое) через conic-gradient, цифры калорий/минут
   • выпить/вино/вечеринка → стеклянная карточка с «бокалом» (треугольник + линия ножки)
     и подписью
   • огонь/горит/сердце жжёт → оранжево-красная карточка с языком пламени формами,
     акцент #ff9500/#ff3b30
   • счастье/солнце/лето → светлая карточка с жёлтым кругом-солнцем и лучами

3. ТИПОГРАФИЧЕСКАЯ ИЕРАРХИЯ ОБЯЗАТЕЛЬНА. В каждом виджете должны быть как минимум
   ТРИ РАЗНЫХ кегля/жирности:
     - гигантский акцент (число/экраниная кнопка/одно слово): 96-180px, 800-900;
     - основной заголовок: 42-48px, 700;
     - подпись/мета: 22-28px, 500-600, вторичный цвет (rgba(...,0.65));
     - мелкая капс-подпись в акценте: 16-20px, 700, uppercase, letter-spacing:0.06em.
   Никогда не делай только один размер текста — это сразу выглядит как дешёвая заглушка.

4. ЦВЕТ. Не ограничивайся одной серостью. Используй:
   - акцентный цвет `accent` как главный brand-color виджета (кнопки, подсветка,
     заголовки, прогресс-кольцо);
   - дополнительные цвета: системная палитра iOS (#ff3b30, #ff9500, #ffcc00, #34c759,
     #5ac8fa, #007aff, #5856d6, #af52de, #ff2d55, #ff6482) — выбери 1-2 подходящих
     к настроению строки;
   - ВСЕГДА есть как минимум один яркий градиент или цветной круг (кнопка, аватар,
     обложка, прогресс) — не оставляй карточку серой.

5. СТЕКЛО И ОБЪЁМ. Обязательно используй:
   - backdrop-filter blur(40px) saturate(180%) на обёртке;
   - inset 0 1px 0 rgba(255,255,255,.35) для блика сверху;
   - внешняя тень 0 30px 80px rgba(0,0,0,.45);
   - цветные «шайбы» (кружки 80-120px с радиальным градиентом от accent к прозрачному
     сверху/снизу, имитируют жидкое стекло iOS 26 — эффект «стеклянной сферы»).
   - хотя бы ОДИН элемент с глазковым бликом (radial-gradient круг белого цвета
     60-80px в углу элемента).

6. АНИМАЦИЯ ВНУТРИ ВИДЖЕТА. Добавь ОДНУ короткую зацикленную CSS-анимацию
   на ключевой элемент (pulse/sway/float/glow), ТОЛЬКО через @keyframes на
   transform/opacity (не меняй layout). Например, пульсирующая кнопка приёма вызова,
   покачивающаяся стрелка поворота, светящаяся иконка Face ID, растущий прогресс,
   крутящийся индикатор загрузки. Длительность 1.4-2.2s, ease-in-out, infinite.
   НЕ анимируй всё подряд — только один акцентный элемент.

7. В карточке НИКОГДА не должно быть текста песни (это выводится отдельно поверх).
   Только подписи/метки/числа/названия UI-элементов ('входящий', 'ответить',
   'отклонить', 'сейчас', 'Apple Pay', '12 мин', 'Face ID', 'iMessage', 'A'/'M'/ и т.п.).

8. НЕ используй <img>, <svg> (любое), внешние шрифты/URL, JS. Только div/span,
   border, border-radius, радиальный/линейный/conic-градиенты, box-shadow,
   transform, opacity, @keyframes. ВСЕ «иконки» собирай из простых фигур
   (круги/квадраты/линии/буквы/стрелки через rotate + клин).

═══════════════════════════════════════════════════════════════
ОСТАЛЬНЫЕ ПОЛЯ:
═══════════════════════════════════════════════════════════════
* theme:      "dark" | "light". Тёмная тема — для ночных/грустных/интимных строк;
              светлая — для радостных/летних/дневных; следи за контрастом.
* accent:     яркий hex из iOS-палитры (#0a84ff/#ff375f/#30d158/#bf5af2/#ff9500/#ffcc00/#64d2ff/#ff3b30).
* background: {"type":"gradient"|"rain"|"sunset"|"light"|"solid", "colors":[hex,hex]}
              — большой фон всего кадра ПОД карточкой, выбирай под настроение:
              gradient — глубокий тёмный, sunset — розово-оранжевый вечер,
              rain — сине-серый дождливый, light — светло-кремовый/белый дневной,
              solid — сплошной акцентный приглушённый.
* entrance:   "rise" (крупные центральные виджеты) | "slide" (списки/уведомления)
              | "fade" (атмосферные карточки) | "pop" (кнопки/реакции).

Возвращай ТОЛЬКО JSON-МАССИВ. Никаких ```, никаких комментариев, никакого текста до или после.
ПОМНИ: качество — это плотность иерархии, узнаваемость как реальный UI iOS,
яркий акцент, стекло с объёмом, минимум 3 уровня шрифта и хотя бы один градиентный акцентный элемент.
"""

# Follow-up prompt for subsequent batches — keeps the model producing the
# same rich quality without re-reading the full system prompt (saves tokens,
# but still explicitly reminds it of the quality bar).
_DIRECTOR_FOLLOWUP_HINT = """\
Продолжение той же песни. Новая порция строк.
ВАЖНО: делай виджеты ТАКОГО ЖЕ ВЫСОКОГО КАЧЕСТВА — трёхслойная структура
(шайба/иконка сверху, насыщенный центр, нижняя строка с кнопками), минимум
3 размера шрифта, стеклянный фон с backdrop-filter blur(40px), цветной акцент,
объёмные тени, одна пульсирующая/светящаяся анимация на акцентном элементе,
УЗНАВАЕМЫЕ как реальные iOS-экраны (звонок, Face ID, Apple Pay, Погода, Карты,
iMessage, Now Playing, Камера, Локскрин, Активность), не повторяй виджеты
из предыдущих строк. Не вставляй текст песни в карточку.

Ответ — ТОЛЬКО JSON-МАССИВ той же длины, в том же порядке, с полями
scene_id/widget_html/theme/accent/background/entrance.
"""


# --------------------------------------------------------------------------- #
# Arena call with batching + retries + generous timeout
# --------------------------------------------------------------------------- #

# Rich widgets have more HTML/CSS, so 4 per request keeps each response
# comfortably sized while giving the model enough bandwidth to design
# dense, high-fidelity cards (3 levels of typography, multiple shapes,
# gradients, an @keyframes animation, etc.).
_BATCH_SIZE = 4
# How many times to retry a batch that times out/5xx before giving up.
_BATCH_RETRIES = 2
# Per-request streaming timeout. Richer widgets = bigger responses; 12 minutes
# covers long tracks with slow first-token latency.
_QUEUE_TIMEOUT = 720.0
# Between-batch pause (seconds) to give Arena time to cool down.
_BATCH_PAUSE = 1.5


def _call_batch(client, batch_llm: list[dict], model: str | None) -> list[dict]:
    """Send one batch of scenes using a FRESH single-turn client.complete() call.

    We deliberately do NOT use a single multi-turn client.chat() session across
    batches: after the first few replies the chat history contains several
    thousand tokens of generated widget HTML, which tends to trigger HTTP 500
    on Arena's backend later in the track. A fresh complete() per batch is
    lightweight, avoids history buildup, and still reuses the same Firefox
    session so captcha does not re-appear.
    """
    # Build a self-contained prompt (no reliance on prior chat context).
    # Include a short list of previously-used widget themes via a simple hint
    # so that adjacent batches don't all come back looking identical.
    user_prompt = (
        DIRECTOR_SYSTEM_PROMPT
        + "\n\n---\n\nСцены для оформления (верни ровно столько же объектов, "
        "тем же порядком, с теми же scene_id):\n"
        + json.dumps(batch_llm, ensure_ascii=False, indent=2)
    )
    cfg = GenerationConfig(queue_timeout=_QUEUE_TIMEOUT)
    last_err: Exception | None = None
    for attempt in range(_BATCH_RETRIES + 1):
        try:
            if model:
                text = client.complete(user_prompt, model=model, config=cfg)
            else:
                text = client.complete(user_prompt, config=cfg)
            break
        except QueueTimeoutError as e:
            last_err = e
            print(f"[arena] batch timeout (attempt {attempt+1}/{_BATCH_RETRIES+1}): {e}",
                  flush=True)
            if attempt < _BATCH_RETRIES:
                time.sleep(2.0 + attempt * 2.0)
                continue
            raise
        except Exception as e:
            # Catch HTTP 500 / other transient Arena errors too.
            last_err = e
            msg = str(e)
            print(f"[arena] batch error (attempt {attempt+1}/{_BATCH_RETRIES+1}): {msg!r}",
                  flush=True)
            if attempt < _BATCH_RETRIES:
                # Longer backoff for hard 5xx — wait a bit so Arena can recover.
                time.sleep(3.0 + attempt * 3.0)
                continue
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
            f"{len(out) if isinstance(out, list) else type(out).__name__}; "
            f"preview: {text[:300]!r}"
        )
    return out


def ask_arena(all_scenes: list[dict], model: str | None) -> tuple[list[dict], list[tuple[int, str]]]:
    """Send scenes to Arena in small batches.

    Returns (out_scenes, failures) where failures is a list of
    (batch_start_index, error_message) for batches that could not be processed
    even after retries (those scenes fall back to stubs individually, not the
    whole track).
    """
    failures: list[tuple[int, str]] = []
    if not _ARENA_OK:
        return _stub_scenes(all_scenes), [(-1, "Arena не инициализирована")]

    # LLM-facing payload WITHOUT per-word arrays (saves ~10x tokens).
    llm_scenes = [
        {"scene_id": s["scene_id"], "start": s["start"], "end": s["end"], "line": s["line"]}
        for s in all_scenes
    ]

    options = AutoOptions(fast_model=model is None)
    out_scenes: list[dict] = []
    total = len(all_scenes)
    with auto_direct_client(options, verification_token="auto") as client:
        for i in range(0, total, _BATCH_SIZE):
            batch_full = all_scenes[i : i + _BATCH_SIZE]
            batch_llm  = llm_scenes[i : i + _BATCH_SIZE]
            print(f"[arena] batch {i//_BATCH_SIZE + 1}/{(total + _BATCH_SIZE - 1)//_BATCH_SIZE}: "
                  f"scenes {batch_full[0]['scene_id']}..{batch_full[-1]['scene_id']}",
                  flush=True)
            try:
                batch_out = _call_batch(client, batch_llm, model)
                ok = True
                err_msg = ""
            except Exception as exc:  # noqa: BLE001
                ok = False
                err_msg = str(exc)
                print(f"[arena] batch FAILED after retries: {err_msg!r}", flush=True)
                failures.append((i, err_msg))
                batch_out = [{} for _ in batch_full]

            for s_in, s_out in zip(batch_full, batch_out):
                if (not ok) or (not isinstance(s_out, dict)) or not s_out.get("widget_html"):
                    s_out = _ensure_widget(s_in, s_out if isinstance(s_out, dict) else {})
                # Force-internal fields (LLM must not override timings/words/line)
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
            if i + _BATCH_SIZE < total:
                time.sleep(_BATCH_PAUSE)
    return out_scenes, failures


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

    # 4. Ask Arena for widget_html (batched; each batch is a fresh complete()
    #    call so chat history doesn't balloon and cause 500s deep in the track).
    t1 = time.monotonic()
    arena_err: str | None = None
    stub_mode = False
    try:
        scenes_out, failures = ask_arena(numbered_scenes, model_name or None)
        if failures:
            # At least one batch failed; the others are real widgets.
            stub_mode = True
            sample = failures[0][1][:200]
            arena_err = (
                f"{len(failures)} батч(ей) из "
                f"{(len(numbered_scenes) + _BATCH_SIZE - 1)//_BATCH_SIZE} "
                f"не обработались Arena (stub на этих сценах): {sample}"
            )
    except QueueTimeoutError as e:
        stub_mode = True
        arena_err = (
            f"Превышен таймаут Arena ({_QUEUE_TIMEOUT:.0f}s). Попробуй быструю модель "
            f"или разбей трек на части. ({e})"
        )
        scenes_out = _stub_scenes(numbered_scenes)
    except Exception as exc:  # noqa: BLE001
        stub_mode = True
        arena_err = f"Ошибка Arena: {exc}"
        scenes_out = _stub_scenes(numbered_scenes)
    arena_ms = int((time.monotonic() - t1) * 1000)

    stub_mode = stub_mode or (not _ARENA_OK)
    state_path = STATE_DIR / f"{track_id}.json"
    state_path.write_text(json.dumps({
        "track_id": track_id,
        "audio": f"/static/uploads/{safe_name}",
        "scenes": scenes_out,
        "timings_ms": {"whisper": whisper_ms, "arena": arena_ms},
        "stub_mode": stub_mode,
        "arena_error": arena_err,
    }, ensure_ascii=False), encoding="utf-8")

    return JSONResponse({
        "track_id": track_id,
        "audio_url": f"/static/uploads/{safe_name}",
        "scenes": scenes_out,
        "timings_ms": {"whisper": whisper_ms, "arena": arena_ms},
        "stub_mode": stub_mode,
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
