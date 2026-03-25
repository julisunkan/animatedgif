import os
import io
import math
import random
import sqlite3
import logging
import zipfile
import base64
import urllib.parse
from datetime import datetime
from flask import (Flask, render_template, request, redirect, url_for,
                   send_file, jsonify, g, send_from_directory)
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import imageio
import numpy as np
import httpx

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "sticker-ai-secret-2024")

DATABASE = "database.db"

STYLE_PROMPTS = {
    "cartoon":  "cartoon style, vibrant bold colors, bold outlines, fun, expressive, 2D animated",
    "anime":    "anime style, manga illustration, colorful, expressive eyes, Japanese animation",
    "emoji":    "emoji flat design, simple, bold, cheerful, clean white background",
    "3d":       "3D render, Pixar style, realistic lighting, glossy, high detail, cute",
    "pixel":    "pixel art, retro 8-bit style, colorful, video game character",
}

MOOD_PROMPTS = {
    "funny":  "laughing, hilarious, comedic, joyful, silly expression",
    "angry":  "angry expression, fierce, dramatic, intense, furious",
    "love":   "loving, cute, heart eyes, adorable, sweet, romantic",
    "sad":    "sad expression, teary eyes, melancholic, drooping",
}

MOOD_SYMBOLS = {
    "funny": ["HA", "LOL", "XD"],
    "angry": ["GRR", "RAH", "MAD"],
    "love":  ["<3", "LUV", "AWW"],
    "sad":   [":'(", "BOO", "OOF"],
}

STYLE_PALETTES = {
    "cartoon": {"bg1": (255, 87, 51),  "bg2": (255, 195, 0),  "accent": (255,255,255)},
    "anime":   {"bg1": (255, 182, 193),"bg2": (147, 197, 253),"accent": (255,255,255)},
    "emoji":   {"bg1": (255, 214, 0),  "bg2": (255, 180, 0),  "accent": (40, 40, 40)},
    "3d":      {"bg1": (67, 56, 202),  "bg2": (168, 85, 247), "accent": (255,255,255)},
    "pixel":   {"bg1": (0, 20, 60),    "bg2": (40, 0, 80),    "accent": (0, 255, 128)},
}

MOOD_COLORS = {
    "funny": (255, 220, 0),
    "angry": (220, 50, 50),
    "love":  (255, 100, 150),
    "sad":   (100, 150, 220),
}


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db():
    with app.app_context():
        db = sqlite3.connect(DATABASE)
        db.executescript("""
            CREATE TABLE IF NOT EXISTS stickers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                enhanced_prompt TEXT,
                file_path TEXT,
                webp_path TEXT,
                style TEXT DEFAULT 'cartoon',
                animation_type TEXT DEFAULT 'bounce',
                mood TEXT DEFAULT 'funny',
                top_text TEXT DEFAULT '',
                bottom_text TEXT DEFAULT '',
                hashtags TEXT DEFAULT '',
                likes INTEGER DEFAULT 0,
                views INTEGER DEFAULT 0,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY,
                groq_api_key TEXT,
                image_api_key TEXT,
                image_api_type TEXT DEFAULT 'huggingface'
            );
        """)
        try:
            db.execute("ALTER TABLE settings ADD COLUMN image_api_key TEXT")
        except Exception:
            pass
        try:
            db.execute("ALTER TABLE settings ADD COLUMN image_api_type TEXT DEFAULT 'huggingface'")
        except Exception:
            pass
        db.commit()
        db.close()
        logging.info("Database initialized")


init_db()


def get_settings():
    db = get_db()
    row = db.execute("SELECT * FROM settings WHERE id = 1").fetchone()
    return dict(row) if row else {}


def get_groq_key():
    return get_settings().get("groq_api_key")


def get_image_api():
    s = get_settings()
    return s.get("image_api_key"), s.get("image_api_type", "huggingface")


def enhance_prompt(prompt, mood, style):
    api_key = get_groq_key()
    if not api_key:
        return prompt
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        system = (
            "You are a creative sticker artist. "
            "Enhance the given prompt for AI image generation. "
            "Make it vivid, expressive, and suitable for an animated sticker. "
            "Keep it under 25 words. Return only the enhanced prompt, no explanations."
        )
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": f"Style: {style}, Mood: {mood}. Original: {prompt}"}],
            model="llama-3.3-70b-versatile",
            max_tokens=80,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logging.warning(f"Groq enhancement failed: {e}")
        return prompt


def generate_hashtags(prompt, style, mood):
    api_key = get_groq_key()
    base = [f"#{style}", f"#{mood}", "#sticker", "#animated", "#AISticker"]
    if not api_key:
        return " ".join(base)
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content":
                       f"Generate 5 short trending hashtags for a {mood} {style} sticker about: {prompt}. "
                       "Return only hashtags separated by spaces."}],
            model="llama-3.3-70b-versatile",
            max_tokens=40,
        )
        raw = completion.choices[0].message.content.strip()
        tags = [w if w.startswith("#") else f"#{w}" for w in raw.split()]
        return " ".join(tags[:5])
    except Exception as e:
        logging.warning(f"Hashtag generation failed: {e}")
        return " ".join(base)


def build_image_prompt(user_prompt, style, mood):
    style_hint = STYLE_PROMPTS.get(style, "")
    mood_hint = MOOD_PROMPTS.get(mood, "")
    return (f"{user_prompt}, {mood_hint}, {style_hint}, "
            f"sticker design, white background, centered, full body, high quality, no text")


def fetch_real_image_hf(prompt, api_key, size=512):
    url = ("https://router.huggingface.co/hf-inference/models/"
           "black-forest-labs/FLUX.1-schnell/v1/images/generations")
    payload = {"prompt": prompt, "n": 1}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        b64 = data["data"][0]["b64_json"]
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        return img.resize((size, size), resample=Image.LANCZOS)


def fetch_real_image_together(prompt, api_key, size=512):
    url = "https://api.together.xyz/v1/images/generations"
    payload = {
        "model": "black-forest-labs/FLUX.1-schnell-Free",
        "prompt": prompt,
        "n": 1,
        "width": size,
        "height": size,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        b64 = data["data"][0]["b64_json"]
        if b64:
            img_bytes = base64.b64decode(b64)
        else:
            img_url = data["data"][0]["url"]
            img_bytes = client.get(img_url).content
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        return img.resize((size, size), resample=Image.LANCZOS)


def fetch_real_image_pollinations(prompt, api_key, size=512):
    encoded = urllib.parse.quote(prompt)
    seed = random.randint(1, 99999)
    url = (f"https://image.pollinations.ai/prompt/{encoded}"
           f"?width={size}&height={size}&seed={seed}&nologo=true&key={api_key}")
    with httpx.Client(timeout=60, follow_redirects=True, headers={
        "User-Agent": "StickerAI/1.0",
        "Referer": "https://pollinations.ai/",
    }) as client:
        r = client.get(url)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGBA")
        return img.resize((size, size), resample=Image.LANCZOS)


def fetch_real_image(prompt, size=512):
    api_key, api_type = get_image_api()
    if not api_key:
        return None
    try:
        if api_type == "together":
            return fetch_real_image_together(prompt, api_key, size)
        elif api_type == "pollinations":
            return fetch_real_image_pollinations(prompt, api_key, size)
        else:
            return fetch_real_image_hf(prompt, api_key, size)
    except Exception as e:
        logging.error(f"Real image fetch failed ({api_type}): {e}")
        return None


# ── Pillow fallback sticker art ────────────────────────────────────────────────

def make_gradient(size, c1, c2):
    img = Image.new("RGBA", size)
    draw = ImageDraw.Draw(img)
    w, h = size
    for y in range(h):
        t = y / h
        r = int(c1[0] + (c2[0] - c1[0]) * t)
        gv = int(c1[1] + (c2[1] - c1[1]) * t)
        b = int(c1[2] + (c2[2] - c1[2]) * t)
        draw.line([(0, y), (w, y)], fill=(r, gv, b, 255))
    return img


def draw_face(draw, cx, cy, r, mood, style):
    palette = STYLE_PALETTES.get(style, STYLE_PALETTES["cartoon"])
    mood_color = MOOD_COLORS.get(mood, (255, 220, 0))

    # Head
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 fill=mood_color, outline=(40, 30, 20), width=max(3, r // 20))

    # Eyes
    ew = max(int(r * 0.18), 5)
    eh = max(int(r * 0.22), 6)
    ex1, ex2 = cx - int(r * 0.35), cx + int(r * 0.35)
    ey = cy - int(r * 0.15)

    if mood == "angry":
        # Angry slanted eyes
        draw.polygon([(ex1 - ew, ey - eh // 2), (ex1 + ew, ey + eh),
                      (ex1 + ew, ey - eh // 2)], fill=(10, 10, 10))
        draw.polygon([(ex2 - ew, ey - eh // 2), (ex2 + ew, ey - eh // 2),
                      (ex2 + ew, ey + eh)], fill=(10, 10, 10))
        # Angry eyebrows
        draw.line([(ex1 - ew, ey - eh), (ex1 + ew + 5, ey - eh // 2 - 8)],
                  fill=(40, 20, 0), width=max(3, r // 15))
        draw.line([(ex2 - ew - 5, ey - eh // 2 - 8), (ex2 + ew, ey - eh)],
                  fill=(40, 20, 0), width=max(3, r // 15))
    elif mood == "love":
        # Heart eyes
        heart_r = max(ew, 8)
        for ex in [ex1, ex2]:
            _draw_heart(draw, ex, ey, heart_r, (220, 40, 80))
    elif mood == "sad":
        # Drooping eyes
        draw.ellipse([ex1 - ew, ey, ex1 + ew, ey + eh * 2], fill=(10, 10, 10))
        draw.ellipse([ex2 - ew, ey, ex2 + ew, ey + eh * 2], fill=(10, 10, 10))
        # Sad eyebrows
        draw.arc([ex1 - ew, ey - eh * 2, ex1 + ew, ey - eh // 2],
                 start=200, end=340, fill=(40, 20, 0), width=max(3, r // 15))
        draw.arc([ex2 - ew, ey - eh * 2, ex2 + ew, ey - eh // 2],
                 start=200, end=340, fill=(40, 20, 0), width=max(3, r // 15))
    else:
        # Normal/funny eyes
        draw.ellipse([ex1 - ew, ey - eh // 2, ex1 + ew, ey + eh], fill=(10, 10, 10))
        draw.ellipse([ex2 - ew, ey - eh // 2, ex2 + ew, ey + eh], fill=(10, 10, 10))
        # Shine
        sw = max(4, ew // 3)
        draw.ellipse([ex1 - sw, ey - eh // 2 + 2, ex1, ey - eh // 2 + 2 + sw], fill=(255, 255, 255))
        draw.ellipse([ex2 - sw, ey - eh // 2 + 2, ex2, ey - eh // 2 + 2 + sw], fill=(255, 255, 255))

    # Mouth
    mouth_y = cy + int(r * 0.35)
    mouth_w = int(r * 0.55)
    if mood == "funny":
        draw.arc([cx - mouth_w, mouth_y - int(r * 0.25),
                  cx + mouth_w, mouth_y + int(r * 0.25)],
                 start=0, end=180, fill=(40, 20, 0), width=max(3, r // 20))
        draw.rectangle([cx - mouth_w + 4, mouth_y - 4, cx + mouth_w - 4, mouth_y + 4],
                       fill=mouth_color(mood))
    elif mood == "angry":
        draw.arc([cx - mouth_w, mouth_y, cx + mouth_w, mouth_y + int(r * 0.35)],
                 start=180, end=360, fill=(40, 20, 0), width=max(3, r // 20))
    elif mood == "love":
        draw.arc([cx - mouth_w, mouth_y - int(r * 0.3),
                  cx + mouth_w, mouth_y + int(r * 0.2)],
                 start=0, end=180, fill=(200, 60, 80), width=max(3, r // 20))
    elif mood == "sad":
        draw.arc([cx - mouth_w, mouth_y, cx + mouth_w, mouth_y + int(r * 0.35)],
                 start=180, end=360, fill=(40, 20, 0), width=max(3, r // 20))
        # Tear
        tx, ty = cx + int(r * 0.4), cy + int(r * 0.1)
        draw.ellipse([tx - 5, ty, tx + 5, ty + 12], fill=(100, 180, 255, 200))

    # Cheeks for love/funny
    if mood in ("love", "funny"):
        ck_r = max(int(r * 0.2), 6)
        draw.ellipse([ex1 - ck_r * 2, ey + eh, ex1, ey + eh + ck_r * 2],
                     fill=(255, 150, 150, 140))
        draw.ellipse([ex2, ey + eh, ex2 + ck_r * 2, ey + eh + ck_r * 2],
                     fill=(255, 150, 150, 140))


def mouth_color(mood):
    return {"funny": (220, 60, 60), "angry": (200, 20, 20),
            "love": (255, 80, 100), "sad": (80, 120, 200)}.get(mood, (200, 60, 60))


def _draw_heart(draw, cx, cy, r, color):
    pts = []
    for angle in range(0, 360, 5):
        a = math.radians(angle)
        x = r * (16 * math.sin(a) ** 3) / 16
        y = -r * (13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a)) / 16
        pts.append((cx + x, cy + y))
    if len(pts) > 2:
        draw.polygon(pts, fill=color)


def draw_body(draw, cx, cy, r, style, mood):
    palette = STYLE_PALETTES.get(style, STYLE_PALETTES["cartoon"])
    mood_color = MOOD_COLORS.get(mood, (255, 220, 0))

    body_color = tuple(min(255, int(c * 0.85)) for c in mood_color)
    bw = int(r * 0.75)
    bh = int(r * 1.1)
    by = cy + r - 10

    # Body oval
    draw.ellipse([cx - bw, by, cx + bw, by + bh],
                 fill=body_color, outline=(40, 30, 20), width=max(2, r // 25))

    # Arms
    arm_y = by + int(bh * 0.25)
    if mood == "funny":
        draw.line([(cx - bw, arm_y), (cx - bw - int(r * 0.5), arm_y - int(r * 0.3))],
                  fill=body_color, width=max(8, r // 8))
        draw.line([(cx + bw, arm_y), (cx + bw + int(r * 0.5), arm_y - int(r * 0.3))],
                  fill=body_color, width=max(8, r // 8))
    elif mood == "love":
        draw.line([(cx - bw, arm_y), (cx - bw - int(r * 0.3), arm_y - int(r * 0.4))],
                  fill=body_color, width=max(8, r // 8))
        draw.line([(cx + bw, arm_y), (cx + bw + int(r * 0.3), arm_y - int(r * 0.4))],
                  fill=body_color, width=max(8, r // 8))
        # Heart hands
        _draw_heart(draw, cx - bw - int(r * 0.3), arm_y - int(r * 0.45),
                    max(r // 6, 8), (255, 80, 120))
        _draw_heart(draw, cx + bw + int(r * 0.3), arm_y - int(r * 0.45),
                    max(r // 6, 8), (255, 80, 120))
    elif mood == "angry":
        draw.line([(cx - bw, arm_y), (cx - bw - int(r * 0.5), arm_y + int(r * 0.2))],
                  fill=body_color, width=max(10, r // 7))
        draw.line([(cx + bw, arm_y), (cx + bw + int(r * 0.5), arm_y + int(r * 0.2))],
                  fill=body_color, width=max(10, r // 7))
    else:
        draw.line([(cx - bw, arm_y), (cx - bw - int(r * 0.35), arm_y + int(r * 0.3))],
                  fill=body_color, width=max(8, r // 8))
        draw.line([(cx + bw, arm_y), (cx + bw + int(r * 0.35), arm_y + int(r * 0.3))],
                  fill=body_color, width=max(8, r // 8))

    # Legs
    leg_y = by + bh - 10
    leg_w = max(8, r // 8)
    draw.rectangle([cx - bw + 15, leg_y, cx - bw + 15 + leg_w, leg_y + int(r * 0.5)],
                   fill=body_color, outline=(40, 30, 20))
    draw.rectangle([cx + bw - 15 - leg_w, leg_y, cx + bw - 15, leg_y + int(r * 0.5)],
                   fill=body_color, outline=(40, 30, 20))

    # Shoes
    shoe_y = leg_y + int(r * 0.5) - 5
    shoe_r = max(int(r * 0.2), 8)
    draw.ellipse([cx - bw + 5, shoe_y, cx - bw + 5 + shoe_r * 2, shoe_y + shoe_r],
                 fill=(40, 30, 20))
    draw.ellipse([cx + bw - 5 - shoe_r * 2, shoe_y, cx + bw - 5, shoe_y + shoe_r],
                 fill=(40, 30, 20))


def draw_accessories(draw, cx, cy, r, style, mood):
    if style == "pixel":
        # Pixel hat
        hw = int(r * 0.9)
        hh = int(r * 0.35)
        hy = cy - r - hh + 4
        step = max(4, hw // 8)
        for px in range(cx - hw // 2, cx + hw // 2, step):
            for py in range(hy, hy + hh, step):
                color = random.choice([(0, 200, 100), (0, 180, 80), (0, 160, 60)])
                draw.rectangle([px, py, px + step - 1, py + step - 1], fill=color)
    elif style == "anime":
        # Anime hair
        hair_pts = [
            (cx - r, cy - r + 10),
            (cx - r - 20, cy - r - 30),
            (cx - r + 20, cy - r - 50),
            (cx, cy - r - 60),
            (cx + r - 20, cy - r - 50),
            (cx + r + 20, cy - r - 30),
            (cx + r, cy - r + 10),
        ]
        mood_color = MOOD_COLORS.get(mood, (255, 220, 0))
        hair_color = tuple(min(255, int(c * 0.6)) for c in mood_color)
        draw.polygon(hair_pts, fill=hair_color, outline=(40, 30, 20))
    elif style == "cartoon":
        # Cartoon hat
        hat_color = MOOD_COLORS.get(mood, (200, 50, 50))
        hat_color = tuple(min(255, int(c * 0.7)) for c in hat_color)
        brim_y = cy - r - 5
        draw.ellipse([cx - int(r * 0.85), brim_y - 12, cx + int(r * 0.85), brim_y + 12],
                     fill=hat_color, outline=(30, 20, 10), width=2)
        draw.rectangle([cx - int(r * 0.55), brim_y - 12 - int(r * 0.6),
                        cx + int(r * 0.55), brim_y - 12 + 5],
                       fill=hat_color, outline=(30, 20, 10), width=2)
    elif style == "3d":
        # 3D glow ring
        for gw in range(8, 0, -2):
            alpha = gw * 15
            draw.arc([cx - r - gw, cy - r - gw, cx + r + gw, cy + r + gw],
                     start=0, end=360,
                     fill=(160, 80, 255, alpha), width=gw)
    elif style == "emoji":
        # Stars around
        for angle_deg in [30, 90, 150, 210, 270, 330]:
            a = math.radians(angle_deg)
            sx = int(cx + (r + 25) * math.cos(a))
            sy = int(cy + (r + 25) * math.sin(a))
            sr = max(6, r // 12)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(255, 230, 50, 220))


def generate_pillow_sticker(prompt, style, mood, size=512):
    palette = STYLE_PALETTES.get(style, STYLE_PALETTES["cartoon"])
    img = make_gradient((size, size), palette["bg1"], palette["bg2"])
    draw = ImageDraw.Draw(img, "RGBA")

    # Decorative background elements
    if style == "pixel":
        step = 32
        for rx in range(0, size, step):
            for ry in range(0, size, step):
                shade = random.choice([(0, 40, 80, 60), (0, 60, 100, 60), (40, 0, 80, 60)])
                draw.rectangle([rx, ry, rx + step - 2, ry + step - 2], fill=shade)
    elif style == "anime":
        for _ in range(18):
            sx = random.randint(10, size - 10)
            sy = random.randint(10, size - 10)
            sr = random.randint(4, 14)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(255, 255, 255, 70))
    elif style == "3d":
        for i in range(4):
            gr = size // 2 + i * 20
            draw.arc([size // 2 - gr, size // 2 - gr, size // 2 + gr, size // 2 + gr],
                     start=0, end=360, fill=(255, 255, 255, 20), width=2)
    elif style == "emoji":
        for angle_deg in range(0, 360, 30):
            a = math.radians(angle_deg)
            sx = int(size // 2 + (size // 2 - 20) * math.cos(a))
            sy = int(size // 2 + (size // 2 - 20) * math.sin(a))
            draw.line([(size // 2, size // 2), (sx, sy)], fill=(255, 255, 255, 25), width=2)

    cx, cy = size // 2, int(size * 0.38)
    r = int(size * 0.19)

    # Body
    draw_body(draw, cx, cy, r, style, mood)

    # Accessories (hat, hair etc)
    draw_accessories(draw, cx, cy, r, style, mood)

    # Face
    draw_face(draw, cx, cy, r, mood, style)


    if style == "pixel":
        # Pixelate
        sm = img.resize((size // 8, size // 8), resample=Image.NEAREST)
        img = sm.resize((size, size), resample=Image.NEAREST)

    return img




def apply_animation(base_img, anim_type, frame_count=14):
    frames = []
    size = base_img.size

    for i in range(frame_count):
        t = i / frame_count
        canvas = Image.new("RGBA", size, (255, 255, 255, 0))

        if anim_type == "bounce":
            offset_y = int(math.sin(t * 2 * math.pi) * 28)
            canvas.paste(base_img, (0, offset_y), base_img)

        elif anim_type == "spin":
            angle = t * 360
            rotated = base_img.rotate(angle, expand=False, resample=Image.BICUBIC)
            canvas.paste(rotated, (0, 0), rotated)

        elif anim_type == "shake":
            ox = int(math.sin(t * 6 * math.pi) * 16)
            oy = int(math.cos(t * 5 * math.pi) * 8)
            canvas.paste(base_img, (ox, oy), base_img)

        elif anim_type == "zoom":
            scale = 0.78 + 0.32 * abs(math.sin(t * math.pi))
            new_w = max(1, int(size[0] * scale))
            new_h = max(1, int(size[1] * scale))
            scaled = base_img.resize((new_w, new_h), resample=Image.LANCZOS)
            px = (size[0] - new_w) // 2
            py = (size[1] - new_h) // 2
            canvas.paste(scaled, (px, py), scaled)

        elif anim_type == "dance":
            angle = math.sin(t * 2 * math.pi) * 14
            offset_y = int(abs(math.sin(t * 2 * math.pi)) * -22)
            offset_x = int(math.sin(t * 4 * math.pi) * 10)
            rotated = base_img.rotate(angle, expand=False, resample=Image.BICUBIC)
            canvas.paste(rotated, (offset_x, offset_y), rotated)

        else:
            canvas.paste(base_img, (0, 0), base_img)

        frames.append(canvas.convert("RGB"))

    return frames


def create_sticker(prompt, style="cartoon", anim_type="bounce", mood="funny"):
    size = 512

    base_img = fetch_real_image(build_image_prompt(prompt, style, mood), size)
    used_real = base_img is not None

    if base_img is None:
        base_img = generate_pillow_sticker(prompt, style, mood, size)
    else:
        enhancer = ImageEnhance.Color(base_img.convert("RGB"))
        base_img = enhancer.enhance(1.2).convert("RGBA")

    frames = apply_animation(base_img, anim_type, frame_count=14)

    os.makedirs("static/stickers", exist_ok=True)
    os.makedirs("static/webp", exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    gif_path = f"static/stickers/sticker_{ts}.gif"
    webp_path = f"static/webp/sticker_{ts}.webp"

    imageio.mimsave(gif_path, [np.array(f) for f in frames], duration=0.07, loop=0)

    webp_frames = [Image.fromarray(np.array(f)).convert("RGBA") for f in frames]
    webp_frames[0].save(
        webp_path, save_all=True, append_images=webp_frames[1:],
        duration=70, loop=0, format="WEBP",
    )

    return gif_path, webp_path, used_real


@app.route("/service-worker.js")
def service_worker():
    return send_from_directory("static", "service-worker.js", mimetype="application/javascript")


@app.route("/manifest.json")
def manifest():
    return send_from_directory("static", "manifest.json", mimetype="application/json")


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    _, api_type = get_image_api()
    has_image_key = bool(get_settings().get("image_api_key"))

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        style = request.form.get("style", "cartoon")
        anim_type = request.form.get("animation", "bounce")
        mood = request.form.get("mood", "funny")
        if not prompt:
            error = "Please enter a prompt!"
            return render_template("index.html", error=error, has_image_key=has_image_key)

        enhanced = enhance_prompt(prompt, mood, style)
        hashtags = generate_hashtags(prompt, style, mood)

        try:
            gif_path, webp_path, used_real = create_sticker(
                enhanced or prompt, style, anim_type, mood
            )
        except Exception as e:
            logging.error(f"Sticker creation failed: {e}")
            error = f"Generation failed: {str(e)}"
            return render_template("index.html", error=error, has_image_key=has_image_key)

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db = get_db()
        cur = db.execute(
            """INSERT INTO stickers
               (prompt, enhanced_prompt, file_path, webp_path, style, animation_type,
                mood, hashtags, likes, views, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?)""",
            (prompt, enhanced, gif_path, webp_path, style, anim_type,
             mood, hashtags, created_at),
        )
        db.commit()
        return redirect(url_for("result", sticker_id=cur.lastrowid, real=int(used_real)))

    return render_template("index.html", error=error, has_image_key=has_image_key)


@app.route("/result/<int:sticker_id>")
def result(sticker_id):
    db = get_db()
    db.execute("UPDATE stickers SET views = views + 1 WHERE id = ?", (sticker_id,))
    db.commit()
    sticker = db.execute("SELECT * FROM stickers WHERE id = ?", (sticker_id,)).fetchone()
    if not sticker:
        return redirect(url_for("index"))
    used_real = request.args.get("real", "0") == "1"
    return render_template("result.html", sticker=sticker, used_real=used_real)


@app.route("/gallery")
def gallery():
    style = request.args.get("style", "")
    mood = request.args.get("mood", "")
    sort = request.args.get("sort", "new")
    search = request.args.get("search", "").strip()

    db = get_db()
    query = "SELECT * FROM stickers WHERE 1=1"
    params = []
    if style:
        query += " AND style = ?"
        params.append(style)
    if mood:
        query += " AND mood = ?"
        params.append(mood)
    if search:
        query += " AND (prompt LIKE ? OR enhanced_prompt LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])

    if sort == "popular":
        query += " ORDER BY (likes + views) DESC"
    elif sort == "likes":
        query += " ORDER BY likes DESC"
    elif sort == "trending":
        query += " ORDER BY (likes * 3 + views) DESC"
    else:
        query += " ORDER BY created_at DESC"

    stickers = db.execute(query, params).fetchall()
    trending = db.execute(
        "SELECT * FROM stickers ORDER BY (likes * 3 + views) DESC LIMIT 6"
    ).fetchall()
    return render_template("gallery.html", stickers=stickers, trending=trending,
                           style=style, mood=mood, sort=sort, search=search)


@app.route("/like/<int:sticker_id>", methods=["POST"])
def like_sticker(sticker_id):
    db = get_db()
    db.execute("UPDATE stickers SET likes = likes + 1 WHERE id = ?", (sticker_id,))
    db.commit()
    row = db.execute("SELECT likes FROM stickers WHERE id = ?", (sticker_id,)).fetchone()
    return jsonify({"likes": row["likes"] if row else 0})


@app.route("/download/<int:sticker_id>/<fmt>")
def download(sticker_id, fmt):
    db = get_db()
    sticker = db.execute("SELECT * FROM stickers WHERE id = ?", (sticker_id,)).fetchone()
    if not sticker:
        return redirect(url_for("gallery"))
    path = sticker["file_path"] if fmt == "gif" else sticker["webp_path"]
    mime = "image/gif" if fmt == "gif" else "image/webp"
    return send_file(path, as_attachment=True,
                     download_name=f"sticker_{sticker_id}.{fmt}", mimetype=mime)


@app.route("/export-pack", methods=["POST"])
def export_pack():
    ids = request.form.getlist("sticker_ids")
    if not ids:
        return redirect(url_for("gallery"))
    db = get_db()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for sid in ids:
            sticker = db.execute("SELECT * FROM stickers WHERE id = ?", (sid,)).fetchone()
            if sticker:
                for p, name in [(sticker["webp_path"], f"sticker_{sid}.webp"),
                                (sticker["file_path"], f"sticker_{sid}.gif")]:
                    if p and os.path.exists(p):
                        zf.write(p, name)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="sticker_pack.zip",
                     mimetype="application/zip")


@app.route("/admin", methods=["GET", "POST"])
def admin():
    db = get_db()
    message = None
    msg_type = None

    if request.method == "POST":
        action = request.form.get("action")
        if action == "save_keys":
            groq_key = request.form.get("groq_api_key", "").strip()
            img_key = request.form.get("image_api_key", "").strip()
            img_type = request.form.get("image_api_type", "huggingface")
            existing = db.execute("SELECT id FROM settings WHERE id = 1").fetchone()
            if existing:
                db.execute(
                    "UPDATE settings SET groq_api_key=?, image_api_key=?, image_api_type=? WHERE id=1",
                    (groq_key or None, img_key or None, img_type)
                )
            else:
                db.execute(
                    "INSERT INTO settings (id, groq_api_key, image_api_key, image_api_type) VALUES (1,?,?,?)",
                    (groq_key or None, img_key or None, img_type)
                )
            db.commit()
            message = "Settings saved!"
            msg_type = "success"
        elif action == "delete_sticker":
            sid = request.form.get("sticker_id")
            sticker = db.execute("SELECT * FROM stickers WHERE id = ?", (sid,)).fetchone()
            if sticker:
                for p in [sticker["file_path"], sticker["webp_path"]]:
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                db.execute("DELETE FROM stickers WHERE id = ?", (sid,))
                db.commit()
            message = "Sticker deleted."
            msg_type = "success"

    stickers = db.execute("SELECT * FROM stickers ORDER BY created_at DESC").fetchall()
    total = len(stickers)
    total_likes = sum(s["likes"] for s in stickers)
    total_views = sum(s["views"] for s in stickers)

    s = get_settings()
    groq_key = s.get("groq_api_key", "")
    img_key = s.get("image_api_key", "")
    img_type = s.get("image_api_type", "huggingface")

    def mask(k):
        if not k:
            return None
        return "*" * (len(k) - 4) + k[-4:] if len(k) > 4 else "****"

    return render_template("admin.html", stickers=stickers, message=message,
                           msg_type=msg_type, total=total, total_likes=total_likes,
                           total_views=total_views,
                           masked_groq=mask(groq_key),
                           masked_img=mask(img_key),
                           img_type=img_type)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
