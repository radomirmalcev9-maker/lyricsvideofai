# lyricsvideofai

Генератор вертикальных (9:16) lyric-видео в стиле **Apple/iOS Motion Lyrics** —
из обычного MP3-трека за один клик: транскрибация через `faster-whisper`, подбор
iOS-виджетов под каждую строку через Arena AI, живой предпросмотр и экспорт в MP4.

## Быстрый старт

**Windows** — дважды кликните `my-lyrics-app/start.bat`.
Скрипт сам создаст виртуальное окружение, поставит все зависимости, установит
Chromium для Playwright и откроет интерфейс в браузере на
<http://127.0.0.1:4770/static/index.html>.

**Linux / macOS:**
```bash
cd my-lyrics-app
chmod +x start.sh
./start.sh
```

## Структура

```
.
├── my-lyrics-app/               # Само приложение
│   ├── server.py                # FastAPI: /api/generate (Whisper + Arena)
│   ├── requirements.txt
│   ├── start.bat / start.sh     # Одно-клик-лаунчеры
│   └── static/
│       └── index.html           # UI 9:16, glassmorphism-виджеты, экспорт MP4
└── arenaaispofandpars/          # Копия arena-api (Python-клиент Arena.ai)
```

## Как работает

1. Загружается MP3 → сохраняется в `static/uploads/`.
2. `faster-whisper` нарезает трек на сегменты с `start/end/text`.
3. Сегменты отправляются в Arena Direct Chat (через `arena_api.auto_direct_client`
   с `verification_token="auto"`) с системным промптом арт-директора, который для
   каждой строки подбирает iOS-виджет и параметры: `face_id`, `battery`,
   `share_location`, `maps`, `notes`, `voice` (Siri), `contacts`, `reaction`,
   `imessage`, `weather`, `player`, `palette`+кошелёк и др., плюс акцентный цвет
   и фон под настроение строки.
4. Фронтенд рендерит стек: статус-бар iOS → кинетическая типографика →
   glassmorphism-виджет с 3D-наклоном → мини-плеер с обложкой и waveform.
5. Кнопка «Экспорт MP4» записывает canvas 1080×1920 @ 60 fps через
   `MediaRecorder`, смешивая с аудиопотоком из Web Audio.

## Возможности UI

- Драг-н-дроп MP3, выбор модели LLM (fast/max/Claude/GPT/Gemini).
- Клики по сетке сцен перематывают аудио на нужный тайминг.
- 18 встроенных iOS-виджетов, фон-дождь с анимированными каплями/луной/фонарём,
  светлая тема для бытовых сцен, пульсирующий Siri-orb, бьющееся сердце и т.д.
- Синхронная анимация текста: предыдущая строка приглушается, акцентные слова
  подсвечиваются цветом виджета.
