# AI Animated Sticker Generator

A production-ready Progressive Web App (PWA) for generating AI-powered animated stickers.

## Stack
- **Backend**: Python Flask
- **Database**: SQLite (database.db)
- **AI**: Groq API (llama-3.3-70b-versatile) for prompt enhancement and hashtag generation
- **Image Processing**: Pillow + imageio for animated GIF/WebP generation
- **Frontend**: Vanilla HTML/CSS/JS with Bootstrap Icons + Font Awesome

## Features
- AI sticker generation with 5 styles (cartoon, anime, emoji, 3D, pixel)
- 5 animation types (bounce, spin, shake, zoom, dance)
- 4 moods (funny, angry, love, sad)
- Meme text overlay (top/bottom)
- Voice input via Web Speech API
- Download as GIF or WebP (WhatsApp format)
- Sticker pack export as ZIP
- Public gallery with likes, views, search, and filter
- Trending algorithm
- Admin panel at /admin (no auth required)
- Full PWA support: manifest.json, service worker, install prompt

## Project Structure
- `app.py` — Flask app, routes, sticker generation logic
- `main.py` — Entry point
- `database.db` — SQLite database
- `templates/` — Jinja2 HTML templates
- `static/css/style.css` — Dark mode stylesheet
- `static/js/app.js` — PWA and interactive JS
- `static/manifest.json` — PWA manifest
- `static/service-worker.js` — Offline caching
- `static/icons/` — App icons (192, 512, maskable)
- `static/stickers/` — Generated GIF files
- `static/webp/` — Generated WebP files

## Running
```bash
python3 -m gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
```

## Configuration
Set Groq API key via Admin panel at `/admin`. Without a key, the app still works but uses unenhanced prompts.
