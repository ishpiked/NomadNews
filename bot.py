import logging
import os
import re
import json
import time
import random
import sqlite3
import asyncio
import httpx
from io import BytesIO

import requests
import feedparser
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont

from telegram import Update, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    CallbackQueryHandler,
    ConversationHandler,
    filters,
)
from telegram.request import HTTPXRequest

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("newsbot")

TOKEN = None  # Replace with your Telegram bot token
OPENROUTER_API_KEY = None  # Replace with your OpenRouter API key
OWNER_ID = None  # Replace with your Telegram user ID for owner commands

MODELS = [
    "minimax/minimax-m3:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "z-ai/glm-5.2:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "minimax/minimax-m2.7:free",
    "poolside/laguna-s-2.1:free",
    "cohere/north-mini-code:free",
    "poolside/laguna-xs-2.1:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

CONFIG_FILE = "config.json"
DB_FILE = "news.db"
MAX_CHANNELS = 4

SOURCES = [
    {"name": "BBC", "rss": "http://feeds.bbci.co.uk/news/rss.xml"},
    {"name": "The Guardian", "rss": "https://www.theguardian.com/world/rss"},
    {"name": "NPR", "rss": "https://feeds.npr.org/1001/rss.xml"},
]

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
SESSION = requests.Session()
SESSION.headers.update(UA)

HTTP = None
SUMMARY_CACHE = {}
HEAD_CACHE = {"items": None, "ts": 0}
HEAD_TTL = 90

httpx_request = HTTPXRequest(connection_pool_size=10, connect_timeout=20, read_timeout=60)
application = None


def _db():
    conn = sqlite3.connect(DB_FILE, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
    except Exception:
        pass
    return conn


def init_db():
    conn = _db()
    conn.execute("DROP TABLE IF EXISTS posts")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen (
            link TEXT PRIMARY KEY,
            title_norm TEXT,
            source TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_title ON seen(title_norm)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS posted (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel TEXT,
            message_id INTEGER,
            link TEXT,
            title TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()
    conn.close()


def normalize_title(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def crumb(*parts):
    return "<b>" + esc(", ".join(str(p) for p in parts)) + "</b>"


def already_posted(link, title):
    conn = _db()
    row = conn.execute(
        "SELECT 1 FROM seen WHERE link = ? OR title_norm = ?",
        (link, normalize_title(title)),
    ).fetchone()
    conn.close()
    return row is not None


def mark_seen(link, title, source):
    conn = _db()
    conn.execute(
        "INSERT OR IGNORE INTO seen (link, title_norm, source) VALUES (?,?,?)",
        (link, normalize_title(title), source),
    )
    conn.commit()
    conn.close()


def save_posted(channel, message_id, link, title):
    conn = _db()
    conn.execute(
        "INSERT INTO posted (channel, message_id, link, title) VALUES (?,?,?,?)",
        (channel, message_id, link, title),
    )
    conn.commit()
    conn.close()


def get_latest_posted():
    conn = _db()
    row = conn.execute(
        "SELECT channel, message_id FROM posted ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {"channels": [], "interval_minutes": 30, "paused": False}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {"channels": [], "interval_minutes": 30, "paused": False}
    cfg.setdefault("channels", [])
    cfg.setdefault("interval_minutes", 30)
    cfg.setdefault("paused", False)
    norm = []
    default_int = cfg["interval_minutes"]
    for c in cfg["channels"]:
        if isinstance(c, str):
            c = {"username": c, "watermark": "@nomadry",
                 "interval": default_int, "paused": False, "last_posted": 0}
        else:
            c.setdefault("username", c.get("channel") or c.get("username"))
            c.setdefault("watermark", "@nomadry")
            c.setdefault("interval", default_int)
            c.setdefault("paused", False)
            c.setdefault("last_posted", 0)
        if c.get("username"):
            norm.append(c)
    cfg["channels"] = norm
    return cfg


def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


async def summarize(text):
    text = text[:4000]
    key = normalize_title(text[:200])
    if key in SUMMARY_CACHE:
        return SUMMARY_CACHE[key]
    system_prompt = (
        "You are a news summarizer. Read the article and write a neutral summary of "
        "EXACTLY 2 to 4 sentences in English. Rules you MUST follow:\n"
        "- Output ONLY the summary sentences, nothing else.\n"
        "- Do NOT repeat the headline.\n"
        "- Do NOT label paragraphs, count characters/words, or add commentary.\n"
        "- Do NOT mention 'the article', 'this summary', 'as an AI', or your instructions.\n"
        "Any output that breaks these rules is a failure."
    )
    payload = {
        "model": MODELS[0],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://t.me",
        "X-Title": "NewsBot",
    }
    for model in MODELS:
        try:
            payload["model"] = model
            r = await HTTP.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
            if r.status_code == 429:
                log.warning("429 on %s, trying next model", model)
                await asyncio.sleep(0.5)
                continue
            if r.status_code != 200:
                log.warning("OpenRouter %s model=%s body=%s", r.status_code, model, r.text[:200])
                await asyncio.sleep(0.5)
                continue
            data = r.json()
            content = data["choices"][0]["message"]["content"].strip()
            out = sanitize_summary(content)
            if out:
                SUMMARY_CACHE[key] = out
            return out
        except Exception as e:
            log.warning("summarize error model=%s: %s", model, e)
            await asyncio.sleep(0.5)
    return None


META_RE = re.compile(
    r"(?i)"
    r"(second|third|first|next|final|another)\s+paragraph|"
    r"count\s+(the\s+)?(characters|words)|let'?s\s+count|"
    r"word\s*count|character\s*count|"
    r"\bparagraph\s*\d|"
    r"here\s+is\s+(a|the)\s+summary|in\s+summary|to\s+summari[sz]e|"
    r"the\s+article\s+(says|states|mentions|reports|discusses|is\s+about)|"
    r"as\s+an\s+ai|i\s+am\s+an\s+ai|you\s+are\s+a|news\s+editor|"
    r"\bnote\s*:|analysis\s*:|commentary\s*:|"
    r"output\s+only|this\s+(summary|article)|"
    r"according\s+to\s+the\s+article|let's\s+break|breaking\s+it\s+down"
)


def sanitize_summary(text):
    if not text:
        return text
    text = text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in sentences:
        if META_RE.search(s):
            break
        kept.append(s.strip())
    out = " ".join(kept).strip()
    if not out:
        lines = [l for l in text.splitlines() if not META_RE.search(l)]
        out = "\n".join(lines).strip()
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def strip_leading_title(text, title):
    if not text or not title:
        return text
    t = text.strip()
    if t.lower().startswith(title.lower()):
        t = t[len(title):].lstrip(" :-\n")
    return t.strip()


async def _fetch_html(url):
    try:
        r = await HTTP.get(url, timeout=20)
        r.raise_for_status()
    except Exception as e:
        log.warning("fetch fail %s: %s", url, e)
        return None
    return r.text


def _extract_text(soup):
    for art in soup.find_all("article"):
        parts = [
            p.get_text(" ", strip=True)
            for p in art.find_all("p")
            if len(p.get_text(strip=True)) > 30
        ]
        t = "\n\n".join(parts).strip()
        if len(t) > 200:
            return t
    main = soup.find("main") or soup
    parts = [
        p.get_text(" ", strip=True)
        for p in main.find_all("p")
        if len(p.get_text(strip=True)) > 40
    ]
    t = "\n\n".join(parts[:50]).strip()
    return t if len(t) > 200 else None


def _extract_image(soup):
    for prop in ("og:image", "og:image:url", "twitter:image"):
        tag = soup.find("meta", property=prop) or soup.find(
            "meta", attrs={"name": prop}
        )
        if tag and tag.get("content"):
            return tag["content"]
    return None


async def fetch_content(url):
    html = await _fetch_html(url)
    if not html:
        return None, None
    soup = BeautifulSoup(html, "html.parser")
    text = _extract_text(soup)
    if not text:
        return None, None
    return text, _extract_image(soup)


async def _fetch_feed(src):
    try:
        r = await HTTP.get(src["rss"], timeout=20)
        r.raise_for_status()
    except Exception as e:
        log.warning("feed fail %s: %s", src["name"], e)
        return []
    f = feedparser.parse(r.text)
    items = []
    for e in f.entries:
        link = (e.get("link") or "").split("?")[0]
        if not link:
            continue
        low = link.lower()
        if any(k in low for k in ("/video", "/videos", "/live", "/audio",
                                   "/gallery", "/podcast")):
            continue
        title = (e.get("title") or "").strip()
        if not title:
            continue
        items.append({"title": title, "link": link, "source": src["name"]})
    return items


async def fetch_headlines():
    if HEAD_CACHE["items"] is not None and (time.time() - HEAD_CACHE["ts"]) < HEAD_TTL:
        return HEAD_CACHE["items"]
    results = await asyncio.gather(*[_fetch_feed(s) for s in SOURCES])
    items = [it for sub in results for it in sub]
    random.shuffle(items)
    HEAD_CACHE["items"] = items
    HEAD_CACHE["ts"] = time.time()
    return items


async def pick_new_article():
    items = await fetch_headlines()
    batch = items[:8]
    fetched = await asyncio.gather(*[fetch_content(it["link"]) for it in batch])
    for it, (text, image) in zip(batch, fetched):
        if already_posted(it["link"], it["title"]):
            continue
        if not text or len(text) < 200:
            continue
        return it, text, image
    return None
    return None


def fit_caption(caption, limit=1024):
    if len(caption) <= limit:
        return caption
    cut = caption[:limit]
    for sep in ("\n\n", "\n", ". "):
        idx = cut.rfind(sep)
        if idx > limit * 0.5:
            return cut[:idx].rstrip() + "…"
    return cut.rstrip() + "…"


FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_PATH = os.path.join(FONT_DIR, "BricolageGrotesque.ttf")
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/bricolagegrotesque/BricolageGrotesque%5Bopsz%2Cwdth%2Cwght%5D.ttf"
FONT_CACHE = {}


def _ensure_bricolage():
    if os.path.exists(FONT_PATH):
        return True
    import glob
    import shutil
    try:
        hits = glob.glob(r"C:\Windows\Fonts\*ricolage*.ttf") + \
               glob.glob(r"C:\Windows\Fonts\*ricolage*.otf")
        if hits:
            os.makedirs(FONT_DIR, exist_ok=True)
            shutil.copy(hits[0], FONT_PATH)
            return True
    except Exception:
        pass
    try:
        os.makedirs(FONT_DIR, exist_ok=True)
        r = requests.get(FONT_URL, timeout=30)
        r.raise_for_status()
        with open(FONT_PATH, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        log.warning("Bricolage font fetch failed: %s", e)
        return False


def get_font(scale):
    if scale in FONT_CACHE:
        return FONT_CACHE[scale]
    font = None
    if _ensure_bricolage():
        try:
            font = ImageFont.truetype(FONT_PATH, scale)
        except Exception:
            font = None
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", scale)
        except Exception:
            font = ImageFont.load_default()
    FONT_CACHE[scale] = font
    return font


async def watermark_image(image_url, text="@nomadry"):
    try:
        r = await HTTP.get(image_url, timeout=25)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGBA")
    except Exception as e:
        log.warning("img fetch fail %s: %s", image_url, e)
        return None
    w, h = img.size
    scale = int(min(w, h) * 0.03)
    scale = max(16, min(scale, 40))
    font = get_font(scale)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x = int(scale * 0.55)
    pad_y = int(scale * 0.4)
    margin = int(scale * 0.6)
    pill_x = w - tw - 2 * pad_x - margin
    pill_y = h - th - 2 * pad_y - margin
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle(
        [pill_x, pill_y, pill_x + tw + 2 * pad_x, pill_y + th + 2 * pad_y],
        radius=(th + 2 * pad_y) // 2,
        fill=(0, 0, 0, 150),
    )
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)
    tx = pill_x + pad_x - bbox[0]
    ty = pill_y + pad_y - bbox[1]
    draw.text((tx + 1, ty + 1), text, font=font, fill=(0, 0, 0, 130))
    draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))
    buf = BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return buf.getvalue()


async def create_post(item, text, image, watermark="@nomadry"):
    summary = await summarize(text)
    if not summary:
        summary = "(Could not summarize this article.)"
    else:
        summary = strip_leading_title(summary, item["title"])
    footer = "\n\n" + esc(watermark)
    body = f"<b>{esc(item['title'])}</b>\n\n{esc(summary)}"
    body = fit_caption(body, 1024 - len(footer))
    caption = body + footer
    image_bytes = None
    if image:
        image_bytes = await watermark_image(image, watermark)
    return caption, image_bytes


async def post_once(context):
    cfg = load_config()
    if cfg.get("paused"):
        return 0
    n = 0
    for ch in cfg["channels"]:
        if ch.get("paused"):
            continue
        if await post_to_channel(context, ch):
            n += 1
    return n


async def post_to_channel(context, ch):
    ready = context.bot_data.setdefault("ready", {})
    cap = ready.get(ch["username"])
    if cap:
        caption, image_bytes, item = cap
        del ready[ch["username"]]
    else:
        result = await pick_new_article()
        if not result:
            return False
        item, text, image = result
        wm = ch.get("watermark") or "@nomadry"
        caption, image_bytes = await create_post(item, text, image, watermark=wm)
    try:
        if image_bytes:
            msg = await context.bot.send_photo(
                chat_id=ch["username"], photo=image_bytes, caption=caption,
                parse_mode="HTML",
            )
        else:
            msg = await context.bot.send_message(
                chat_id=ch["username"], text=caption, parse_mode="HTML",
            )
        save_posted(ch["username"], msg.message_id, item["link"], item["title"])
        mark_seen(item["link"], item["title"], item["source"])
        return True
    except Exception as e:
        log.warning("post to %s failed: %s", ch["username"], e)
        return False


async def ticker(context):
    cfg = load_config()
    if cfg.get("paused"):
        return
    now = time.time()
    dirty = False
    for ch in cfg["channels"]:
        if ch.get("paused"):
            continue
        interval = max(1, ch.get("interval", cfg.get("interval_minutes", 30))) * 60
        if now - ch.get("last_posted", 0) < interval:
            continue
        if await post_to_channel(context, ch):
            ch["last_posted"] = now
            dirty = True
    if dirty:
        save_config(cfg)


async def warmer(context):
    cfg = load_config()
    ready = context.bot_data.setdefault("ready", {})
    for ch in cfg["channels"]:
        if cfg.get("paused") or ch.get("paused"):
            ready.pop(ch["username"], None)
            continue
        if ch["username"] in ready:
            continue
        try:
            result = await pick_new_article()
            if not result:
                continue
            item, text, image = result
            wm = ch.get("watermark") or "@nomadry"
            caption, image_bytes = await create_post(item, text, image, watermark=wm)
            ready[ch["username"]] = (caption, image_bytes, item)
            mark_seen(item["link"], item["title"], item["source"])
        except Exception as e:
            log.warning("warmer error: %s", e)


def schedule_jobs(app):
    for job in app.job_queue.get_jobs_by_name("ticker"):
        job.schedule_removal()
    app.job_queue.run_repeating(ticker, interval=30, first=10, name="ticker")
    app.job_queue.run_repeating(warmer, interval=20, first=5, name="warmer")


async def start_cmd(update, context):
    greeting = (
        "Hello there.\n\n"
        "Welcome to Nomad News Bot.\n\n"
        "This bot delivers the latest news from trusted sources.\n"
        "It covers world news, politics, business, technology, science, sports, and entertainment.\n"
        "Each story is summarized for quick reading.\n"
        "The bot posts automatically to your channels on a schedule you set.\n\n"
        "Tap the button below to get the latest news right now."
    )
    kb = [[InlineKeyboardButton("Latest News", callback_data="unews")]]
    if is_owner(update):
        kb.append([InlineKeyboardButton("Owner Panel", callback_data="m")])
    await update.message.reply_text(greeting, reply_markup=InlineKeyboardMarkup(kb))


async def setchannel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return
    if not context.args:
        await update.message.reply_text("Please provide a channel username. Example: /setchannel @mychannel")
        return
    ch = context.args[0]
    if not ch.startswith("@"):
        ch = "@" + ch
    cfg = load_config()
    if any(c["username"] == ch for c in cfg["channels"]):
        await update.message.reply_text("That channel is already in your list.")
        return
    if len(cfg["channels"]) >= MAX_CHANNELS:
        await update.message.reply_text("You have reached the maximum of four channels.")
        return
    cfg["channels"].append({"username": ch, "watermark": "@nomadry",
                            "interval": cfg["interval_minutes"], "paused": False,
                            "last_posted": 0})
    save_config(cfg)
    await update.message.reply_text(f"Channel <b>{esc(ch)}</b> has been added. You now have {len(cfg['channels'])} channels.", parse_mode="HTML")


async def removechannel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return
    if not context.args:
        await update.message.reply_text("Please provide a channel username to remove. Example: /removechannel @mychannel")
        return
    ch = context.args[0]
    if not ch.startswith("@"):
        ch = "@" + ch
    cfg = load_config()
    if any(c["username"] == ch for c in cfg["channels"]):
        cfg["channels"] = [c for c in cfg["channels"] if c["username"] != ch]
        save_config(cfg)
        await update.message.reply_text(f"Channel <b>{esc(ch)}</b> has been removed from your list.", parse_mode="HTML")
    else:
        await update.message.reply_text("That channel is not in your list.")


async def channels_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = load_config()
    if not cfg["channels"]:
        await update.message.reply_text("You have not added any channels yet.")
    else:
        lines = ["Your configured channels:"]
        for c in cfg["channels"]:
            status = " paused" if c.get("paused") else ""
            lines.append(c["username"] + status)
        await update.message.reply_text("\n".join(lines))


async def setinterval_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Please provide a number of minutes. Example: /setinterval 30")
        return
    mins = max(1, int(context.args[0]))
    cfg = load_config()
    cfg["interval_minutes"] = mins
    save_config(cfg)
    schedule_jobs(context.application)
    await update.message.reply_text(f"The default posting interval has been set to <b>{esc(str(mins))}</b> minutes.", parse_mode="HTML")


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return
    cfg = load_config()
    jobs = context.application.job_queue.get_jobs_by_name("ticker")
    next_t = "not available"
    if jobs:
        try:
            nxt = jobs[0].next_t
            if nxt:
                next_t = nxt.strftime("%H:%M:%S")
        except Exception:
            pass
    lines = [
        "<b>Bot Status</b>",
        "",
        f"<b>Channels configured:</b> {len(cfg['channels'])} of {MAX_CHANNELS}",
        f"<b>Default interval:</b> {cfg['interval_minutes']} minutes",
        f"<b>Global pause:</b> {cfg.get('paused')}",
        f"<b>Next scheduled post:</b> {next_t}",
    ]
    if cfg["channels"]:
        chs = ", ".join(c["username"] + (" paused" if c.get("paused") else "")
                        for c in cfg["channels"])
        lines.append("")
        lines.append(f"<b>Targets:</b> {chs}")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def pause_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return
    cfg = load_config()
    cfg["paused"] = True
    save_config(cfg)
    await update.message.reply_text("Automatic posting has been paused for all channels.")


async def resume_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update):
        return
    cfg = load_config()
    cfg["paused"] = False
    save_config(cfg)
    await update.message.reply_text("Automatic posting has been resumed for all channels.")


async def news_cmd(update, context):
    latest = get_latest_posted()
    if not latest:
        await update.message.reply_text("No news has been posted yet. The bot will post when new articles are available.")
        return
    channel, message_id = latest
    try:
        await context.bot.copy_message(
            chat_id=update.effective_chat.id,
            from_chat_id=channel,
            message_id=message_id,
        )
        return
    except Exception as e:
        log.warning("forward latest news failed: %s", e)
    if not is_owner(update):
        await update.message.reply_text("The latest news could not be retrieved at this time. Please try again later.")
        return
    items = await fetch_headlines()
    for it in items:
        if already_posted(it["link"], it["title"]):
            continue
        text, image = await fetch_content(it["link"])
        if not text:
            continue
        caption, image_bytes = await create_post(it, text, image)
        try:
            if image_bytes:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id, photo=image_bytes,
                    caption=caption, parse_mode="HTML",
                )
            else:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id, text=caption,
                    parse_mode="HTML",
                )
        except Exception as e2:
            log.warning("news fallback failed: %s", e2)
            await update.message.reply_text("The latest news could not be retrieved at this time. Please try again later.")
        return
    await update.message.reply_text("The latest news could not be retrieved at this time. Please try again later.")


def is_owner(update):
    return OWNER_ID is not None and update.effective_user.id == OWNER_ID


def owner_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Channels", callback_data="chlist"),
         InlineKeyboardButton("Add Channel", callback_data="add")],
        [InlineKeyboardButton("Remove Channel", callback_data="rm"),
         InlineKeyboardButton("Set Interval", callback_data="gi")],
        [InlineKeyboardButton("Pause All", callback_data="pa"),
         InlineKeyboardButton("Resume All", callback_data="ra")],
        [InlineKeyboardButton("Post All", callback_data="pn"),
         InlineKeyboardButton("Status", callback_data="st")],
        [InlineKeyboardButton("Help", callback_data="help"),
         InlineKeyboardButton("Close", callback_data="close")],
        [InlineKeyboardButton("Back to Panel", callback_data="m")],
    ])


async def panel_cmd(update, context):
    if not is_owner(update):
        return
    await update.message.reply_text(
        "<b>News Bot</b> Advanced Owner Management &amp; Configuration Panel",
        reply_markup=owner_keyboard(), parse_mode="HTML",
    )


async def _panel_status_text(context):
    cfg = load_config()
    lines = [
        f"<b>Channels:</b> {len(cfg['channels'])}/{MAX_CHANNELS}",
        f"<b>Default interval:</b> {cfg['interval_minutes']} min",
        f"<b>All paused:</b> {cfg.get('paused')}",
    ]
    if cfg["channels"]:
        chs = ", ".join(c["username"] + (" paused" if c.get("paused") else "")
                        for c in cfg["channels"])
        lines.append(f"<b>Targets:</b> {chs}")
    jobs = context.application.job_queue.get_jobs_by_name("ticker")
    if jobs:
        try:
            nxt = jobs[0].next_t
            if nxt:
                lines.append(f"<b>Ticker next:</b> {nxt.strftime('%H:%M:%S')}")
        except Exception:
            pass
    return "\n".join(lines)


def cancel_kb():
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Cancel", callback_data="cancel")]]
    )


def channels_list_keyboard():
    cfg = load_config()
    kb = [[InlineKeyboardButton(c["username"] + (" paused" if c.get("paused") else ""),
                                callback_data="c:" + c["username"])] for c in cfg["channels"]]
    kb.append([InlineKeyboardButton("Back to Panel", callback_data="m")])
    return InlineKeyboardMarkup(kb)


def remove_list_keyboard():
    cfg = load_config()
    kb = [[InlineKeyboardButton(c["username"], callback_data="rm:" + c["username"])]
          for c in cfg["channels"]]
    kb.append([InlineKeyboardButton("Back to Panel", callback_data="m")])
    return InlineKeyboardMarkup(kb)


def channel_detail_keyboard(c):
    uname = c["username"]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Set Watermark", callback_data="cwm:" + uname),
         InlineKeyboardButton("Set Interval", callback_data="cint:" + uname)],
        [InlineKeyboardButton("Pause" if not c.get("paused") else "Resume",
                              callback_data=("cp:" if not c.get("paused") else "cr:") + uname),
         InlineKeyboardButton("Post Now", callback_data="cpn:" + uname)],
        [InlineKeyboardButton("Stats", callback_data="cs:" + uname),
         InlineKeyboardButton("Back to Channels", callback_data="chlist")],
        [InlineKeyboardButton("Back to Panel", callback_data="m")],
    ])


def channel_stats(username):
    conn = _db()
    count = conn.execute("SELECT COUNT(*) FROM posted WHERE channel=?",
                         (username,)).fetchone()[0]
    row = conn.execute(
        "SELECT title, created_at FROM posted WHERE channel=? ORDER BY id DESC LIMIT 1",
        (username,)).fetchone()
    conn.close()
    return {"count": count, "last": row[1] if row else "never",
            "title": row[0] if row else "-"}


def channel_detail_text(c):
    st = channel_stats(c["username"])
    return (f"<b>Watermark:</b> {esc(c.get('watermark') or '@nomadry')}\n"
            f"<b>Interval:</b> {c.get('interval') or 30} min\n"
            f"<b>Paused:</b> {c.get('paused', False)}\n"
            f"<b>Posts:</b> {st['count']}\n"
            f"<b>Last post:</b> {st['last']}")


async def user_cb(update, context):
    q = update.callback_query
    if q.data == "unews":
        chat = update.effective_chat.id
        conn = _db()
        row = conn.execute(
            "SELECT channel, message_id FROM posted ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if not row:
            await q.answer("There is no news available at this moment.")
            return
        try:
            await context.bot.copy_message(
                chat_id=chat, from_chat_id=row[0], message_id=row[1]
            )
            await q.answer()
        except Exception as e:
            log.warning("unews forward failed: %s", e)
            await q.answer("The latest news could not be retrieved at this time.")


async def panel_cb(update, context):
    q = update.callback_query
    await q.answer()
    if not is_owner(update):
        return
    data = q.data
    cfg = load_config()
    if data == "m":
        await q.edit_message_text("<b>News Bot</b> Advanced Owner Management and Configuration Panel", reply_markup=owner_keyboard(), parse_mode="HTML")
    elif data == "st":
        await q.edit_message_text("<b>News Bot</b> Advanced Owner Management and Configuration Panel\n\n" + await _panel_status_text(context),
                                  reply_markup=owner_keyboard(), parse_mode="HTML")
    elif data == "chlist":
        if not cfg["channels"]:
            await q.edit_message_text("You have not added any channels yet. Use the Add Channel button to add one.",
                                     reply_markup=owner_keyboard(), parse_mode="HTML")
            return
        await q.edit_message_text(
            crumb("Owner Panel", "Channels") + "\n\n"
            "Tap a channel below to view details and manage its settings.",
            reply_markup=channels_list_keyboard(), parse_mode="HTML")
    elif data == "pa":
        cfg["paused"] = True; save_config(cfg)
        await q.edit_message_text("<b>News Bot</b> Advanced Owner Management and Configuration Panel\n\nAutomatic posting has been paused for all channels.",
                                  reply_markup=owner_keyboard(), parse_mode="HTML")
    elif data == "ra":
        cfg["paused"] = False; save_config(cfg)
        await q.edit_message_text("<b>News Bot</b> Advanced Owner Management and Configuration Panel\n\nAutomatic posting has been resumed for all channels.",
                                  reply_markup=owner_keyboard(), parse_mode="HTML")
    elif data == "pn":
        await q.answer("Working...")
        n = await post_once(context)
        result = ("Posts have been sent to all channels." if n > 0
                  else "No posts were sent. Channels may be paused or there are no new articles.")
        await q.edit_message_text("<b>News Bot</b> Advanced Owner Management and Configuration Panel\n\n" + result,
                                  reply_markup=owner_keyboard(), parse_mode="HTML")
    elif data == "help":
        await q.edit_message_text(
            crumb("Owner Panel", "Help") + "\n\n"
            "Channels: view and manage your target channels\n"
            "Add Channel: add a new channel by username or ID\n"
            "Remove Channel: remove a channel from your list\n"
            "Set Interval: change the default minutes between posts\n"
            "Pause All: stop automatic posting for all channels\n"
            "Resume All: restart automatic posting for all channels\n"
            "Post All: send a post to every channel right now\n"
            "Per channel: set watermark, set interval, pause, post now, view stats",
            reply_markup=owner_keyboard(), parse_mode="HTML")
    elif data == "close":
        try:
            await q.delete_message()
        except Exception:
            try:
                await q.edit_message_text("The panel has been closed.",
                                          reply_markup=owner_keyboard(), parse_mode="HTML")
            except Exception:
                pass
        return
    elif data == "rm":
        if not cfg["channels"]:
            await q.edit_message_text("There are no channels to remove.", reply_markup=owner_keyboard(), parse_mode="HTML")
            return
        await q.edit_message_text(crumb("Owner Panel", "Remove Channel") + "\n\nTap a channel to remove it.",
                                 reply_markup=remove_list_keyboard(), parse_mode="HTML")
    elif data.startswith("rm:"):
        target = data[3:]
        await q.edit_message_text(
            crumb("Owner Panel", "Remove Channel") + "\n\nAre you sure you want to remove <b>" + esc(target) + "</b>?",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("Confirm", callback_data="rmc:" + target),
                InlineKeyboardButton("Back to Panel", callback_data="m")]]), parse_mode="HTML")
    elif data.startswith("rmc:"):
        target = data[4:]
        cfg["channels"] = [c for c in cfg["channels"] if c["username"] != target]
        save_config(cfg)
        await q.edit_message_text("Channel <b>" + esc(target) + "</b> has been removed.", reply_markup=owner_keyboard(), parse_mode="HTML")
    elif data.startswith("c:"):
        target = data[2:]
        c = next((x for x in cfg["channels"] if x["username"] == target), None)
        if not c:
            await q.edit_message_text("The channel was not found in your list.", reply_markup=owner_keyboard(), parse_mode="HTML")
            return
        await q.edit_message_text(crumb("Owner Panel", "Channel", target) + "\n\n" + channel_detail_text(c),
                                  reply_markup=channel_detail_keyboard(c), parse_mode="HTML")
    elif data.startswith("cp:"):
        target = data[3:]
        c = next((x for x in cfg["channels"] if x["username"] == target), None)
        if not c:
            await q.edit_message_text("The channel was not found in your list.", reply_markup=owner_keyboard(), parse_mode="HTML")
            return
        c["paused"] = True
        save_config(cfg)
        await q.edit_message_text(crumb("Owner Panel", "Channel", target) + "\n\n" + channel_detail_text(c),
                                  reply_markup=channel_detail_keyboard(c), parse_mode="HTML")
    elif data.startswith("cr:"):
        target = data[3:]
        c = next((x for x in cfg["channels"] if x["username"] == target), None)
        if not c:
            await q.edit_message_text("The channel was not found in your list.", reply_markup=owner_keyboard(), parse_mode="HTML")
            return
        c["paused"] = False
        save_config(cfg)
        await q.edit_message_text(crumb("Owner Panel", "Channel", target) + "\n\n" + channel_detail_text(c),
                                  reply_markup=channel_detail_keyboard(c), parse_mode="HTML")
    elif data.startswith("cpn:"):
        target = data[4:]
        c = next((x for x in cfg["channels"] if x["username"] == target), None)
        if not c:
            await q.edit_message_text("The channel was not found in your list.", reply_markup=owner_keyboard(), parse_mode="HTML")
            return
        await q.answer("Working...")
        await post_to_channel(context, c)
        await q.edit_message_text(crumb("Owner Panel", "Channel", target) + "\n\nA post has been sent to <b>" + esc(target) + "</b>.",
                                  reply_markup=channel_detail_keyboard(c), parse_mode="HTML")
    elif data.startswith("cs:"):
        target = data[3:]
        st = channel_stats(target)
        await q.edit_message_text(
            crumb("Owner Panel", "Channel", target, "Stats") + "\n\n"
            f"Total posts: {st['count']}\nLast post time: {st['last']}\n"
            f"Last post title: {esc(st['title'])}",
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("Back to Channel", callback_data="c:" + target)]]), parse_mode="HTML")


AWAIT_INPUT = 1


async def conv_start(update, context):
    q = update.callback_query
    await q.answer()
    if not is_owner(update):
        return ConversationHandler.END
    data = q.data
    action = data.split(":")[0]
    target = data.split(":", 1)[1] if ":" in data else None
    context.user_data["conv_action"] = action
    context.user_data["conv_target"] = target
    context.user_data["conv_chat"] = q.message.chat_id
    context.user_data["conv_msg_id"] = q.message.message_id
    prompts = {
        "add": crumb("Owner Panel", "Add Channel") + "\n\nPlease enter the channel username or numeric ID you want to add. Tap Cancel to go back.",
        "gi": crumb("Owner Panel", "Set Interval") + "\n\nPlease enter the default interval in minutes between posts. Tap Cancel to go back.",
        "cwm": crumb("Owner Panel", "Channel", target, "Watermark") + "\n\nPlease enter the watermark text for this channel. Tap Cancel to go back.",
        "cint": crumb("Owner Panel", "Channel", target, "Interval") + "\n\nPlease enter the posting interval in minutes for this channel. Tap Cancel to go back.",
    }
    await q.edit_message_text(prompts.get(action, "Please enter a value."),
                              reply_markup=cancel_kb(), parse_mode="HTML")
    return AWAIT_INPUT


async def conv_input(update, context):
    action = context.user_data.get("conv_action")
    target = context.user_data.get("conv_target")
    if update.message.text is None:
        await update.message.reply_text("Send a text value, or /cancel.", parse_mode="HTML")
        return AWAIT_INPUT
    val = update.message.text.strip()
    chat = context.user_data.get("conv_chat")
    mid = context.user_data.get("conv_msg_id")

    async def finish(text, keyboard):
        if chat and mid:
            try:
                await context.bot.edit_message_text(
                    text, chat_id=chat, message_id=mid,
                    reply_markup=keyboard, parse_mode="HTML")
            except Exception:
                pass

    # Check for empty input
    if not val:
        await update.message.reply_text("Please enter a value or tap Cancel.", parse_mode="HTML")
        return AWAIT_INPUT

    # Handle add channel action
    if action == "add":
        ch = val if (val.lstrip("-").isdigit() or val.startswith("@")) else "@" + val
        if len(ch) < 2:
            await update.message.reply_text("Please enter a valid channel username or ID. Tap Cancel to go back.", parse_mode="HTML")
            return AWAIT_INPUT
        cfg = load_config()
        if any(c["username"] == ch for c in cfg["channels"]):
            await update.message.reply_text("That channel is already in your list.", parse_mode="HTML")
        elif len(cfg["channels"]) >= MAX_CHANNELS:
            await update.message.reply_text("You have reached the maximum of four channels.", parse_mode="HTML")
        else:
            cfg["channels"].append({"username": ch, "watermark": "@nomadry",
                                    "interval": cfg["interval_minutes"], "paused": False,
                                    "last_posted": 0})
            save_config(cfg)
            await update.message.reply_text(f"Channel <b>{esc(ch)}</b> has been added.", parse_mode="HTML")
        await finish(
            crumb("Owner Panel", "Channels") + "\n\n"
            "Tap a channel below to view details and manage its settings.",
            channels_list_keyboard())

    # Handle global interval setting
    elif action == "gi":
        if not val.isdigit():
            await update.message.reply_text("Please enter a number. Tap Cancel to go back.", parse_mode="HTML")
            return AWAIT_INPUT
        cfg = load_config()
        cfg["interval_minutes"] = max(1, int(val))
        save_config(cfg)
        await update.message.reply_text("The default interval has been set to <b>" + esc(val) + "</b> minutes.", parse_mode="HTML")
        await finish("<b>News Bot</b> Advanced Owner Management and Configuration Panel",
                     owner_keyboard())

    # Handle per-channel watermark setting
    elif action == "cwm":
        cfg = load_config()
        c = next((x for x in cfg["channels"] if x["username"] == target), None)
        if not c:
            await update.message.reply_text("The channel was not found in your list.", parse_mode="HTML")
            await finish(crumb("Owner Panel", "Channels") + "\n\n"
                         "Tap a channel below to view details and manage its settings.",
                         channels_list_keyboard())
        else:
            c["watermark"] = val
            save_config(cfg)
            await update.message.reply_text("The watermark for <b>" + esc(target) + "</b> has been set to: <b>" + esc(val) + "</b>", parse_mode="HTML")
            await finish(crumb("Owner Panel", "Channel", target) + "\n\n" + channel_detail_text(c),
                         channel_detail_keyboard(c))

    # Handle per-channel interval setting
    elif action == "cint":
        if not val.isdigit():
            await update.message.reply_text("Please enter a number. Tap Cancel to go back.", parse_mode="HTML")
            return AWAIT_INPUT
        cfg = load_config()
        c = next((x for x in cfg["channels"] if x["username"] == target), None)
        if not c:
            await update.message.reply_text("The channel was not found in your list.", parse_mode="HTML")
            await finish(crumb("Owner Panel", "Channels") + "\n\n"
                         "Tap a channel below to view details and manage its settings.",
                         channels_list_keyboard())
        else:
            c["interval"] = max(1, int(val))
            save_config(cfg)
            await update.message.reply_text("The interval for <b>" + esc(target) + "</b> has been set to <b>" + esc(val) + "</b> minutes.", parse_mode="HTML")
            await finish(crumb("Owner Panel", "Channel", target) + "\n\n" + channel_detail_text(c),
                         channel_detail_keyboard(c))

    context.user_data.clear()
    return ConversationHandler.END


async def conv_cancel(update, context):
    chat = context.user_data.get("conv_chat")
    mid = context.user_data.get("conv_msg_id")
    if chat and mid:
        try:
            await context.bot.edit_message_text(
                "<b>News Bot</b> Advanced Owner Management &amp; Configuration Panel",
                chat_id=chat, message_id=mid,
                reply_markup=owner_keyboard(), parse_mode="HTML")
        except Exception:
            pass
    await update.message.reply_text("The action has been cancelled.", parse_mode="HTML")
    context.user_data.clear()
    return ConversationHandler.END


async def conv_any_cb(update, context):
    q = update.callback_query
    await q.answer()
    data = q.data
    chat = context.user_data.get("conv_chat")
    mid = context.user_data.get("conv_msg_id")
    context.user_data.clear()
    if data == "close":
        try:
            await q.delete_message()
        except Exception:
            pass
        return ConversationHandler.END
    text = ("The action has been cancelled." if data == "cancel"
            else "<b>News Bot</b> Advanced Owner Management &amp; Configuration Panel")
    if chat and mid:
        try:
            await context.bot.edit_message_text(
                text, chat_id=chat, message_id=mid,
                reply_markup=owner_keyboard(), parse_mode="HTML")
        except Exception:
            pass
    return ConversationHandler.END


async def conv_cancel_cb(update, context):
    await update.callback_query.answer()
    await update.callback_query.edit_message_text("The action has been cancelled.", reply_markup=owner_keyboard())
    return ConversationHandler.END


CONV = ConversationHandler(
    entry_points=[CallbackQueryHandler(conv_start, pattern="^(add|gi|cwm|cint)")],
    states={AWAIT_INPUT: [
        CallbackQueryHandler(conv_any_cb, pattern=".*"),
        MessageHandler(filters.ALL & ~filters.COMMAND, conv_input),
    ]},
    fallbacks=[CommandHandler("cancel", conv_cancel),
               CallbackQueryHandler(conv_cancel_cb, pattern="^cancel$")],
)


async def _shutdown(app):
    global HTTP
    if HTTP:
        await HTTP.aclose()


def main():
    global application, HTTP
    init_db()
    HTTP = httpx.AsyncClient(headers=UA, timeout=30.0, limits=httpx.Limits(max_connections=20, max_keepalive_connections=10))
    application = (
        ApplicationBuilder()
        .token(TOKEN)
        .request(httpx_request)
        .build()
    )
    application.add_handler(CONV)
    application.add_handler(CallbackQueryHandler(
        panel_cb,
        pattern="^(m|st|chlist|rm|rm:.+|rmc:.+|c:.+|cp:.+|cr:.+|cpn:.+|cs:.+|pa|ra|pn|help|close)$",
    ))
    application.add_handler(CallbackQueryHandler(user_cb, pattern="^unews$"))
    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("panel", panel_cmd))
    application.add_handler(CommandHandler("setchannel", setchannel_cmd))
    application.add_handler(CommandHandler("removechannel", removechannel_cmd))
    application.add_handler(CommandHandler("channels", channels_cmd))
    application.add_handler(CommandHandler("setinterval", setinterval_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CommandHandler("pause", pause_cmd))
    application.add_handler(CommandHandler("resume", resume_cmd))
    application.add_handler(CommandHandler("news", news_cmd))
    application.post_shutdown(_shutdown)
    cfg = load_config()
    schedule_jobs(application)
    log.info("bot starting with %d sources, %d channels", len(SOURCES), len(cfg["channels"]))
    application.run_polling()


if __name__ == "__main__":
    main()
