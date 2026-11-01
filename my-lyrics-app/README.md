# Apple Motion Lyrics Generator

Вертикальные 9:16 lyric-видео в стиле Apple/iOS Motion — из MP3 за один клик.

## Стек
- **FastAPI + Uvicorn** — бэкенд.
- **faster-whisper** — транскрибация и тайминги строк.
- **arena_api** (из `arenaaispofandpars/`) — LLM через Arena Direct Chat, которая подбирает iOS-виджет под каждую строку.
- **Чистый HTML/CSS/JS** — фронтенд: статус-бар iOS, glassmorphism-виджеты, 3D-наклон, кинетическая типографика, плеер с обложкой.
- **MP4 1080×1920** — экспорт через `canvas.captureStream()` + `MediaRecorder`.

## Запуск

```bash
cd my-lyrics-app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Для работы Arena-вызова — нужна активная сессия в arena.ai в локальном браузере
# и установленный arena-api с extras:
pip install -e ../arenaaispofandpars/[auto]

python server.py
# Открой http://127.0.0.1:4770/static/index.html
```

Если сессии Arena нет, сервер работает в **stub-режиме** и отдаёт демонстрационные
сцены, чтобы можно было отрабатывать UI и экспорт без входа в Arena.

## Структура проекта
```
my-lyrics-app/
├── server.py                 # FastAPI: /api/generate, /api/health
├── requirements.txt
└── static/
    ├── index.html            # UI, рендер виджетов, экспорт MP4
    ├── uploads/              # загруженные MP3
    └── state/                # кэш сцен по track_id
```

## Поддерживаемые виджеты
`face_id`, `voice` (Siri), `heart_like`, `battery`, `share_location`, `maps`,
`notes`, `photo_note`, `palette` (с кошельком), `stopwatch`, `contacts`,
`reaction` (смайлы), `imessage`, `weather`, `player`, `toggle`, `timer`,
`heart_rate`, плюс фоны `gradient`/`rain`/`light`/`noise` и акцентные цвета.
