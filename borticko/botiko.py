import asyncio
import json
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Set

from aiogram import Bot, Dispatcher, types
from aiogram.types import ChatPermissions
from aiogram.utils import executor

# ================== НАСТРОЙКИ ==================
BLACKLIST_FILE = "bot_ids.json"

# Список запрещённых слов
BAD_WORDS = [
    "дурак", "идиот", "дебил", "урод", "сволочь",
    "сука", "бля", "пизда", "хуй", "мудак",
    "гандон", "редиска", "чмо", "тварь"
]

WARN_LIMIT = 3
SPAM_LIMIT = 5
SPAM_INTERVAL = 10
BOT_BAN_THRESHOLD = 0.7
MANUAL_VERIFICATION_NEEDED = 0.5

FEATURE_WEIGHTS = {
    "username_bot": 0.3,
    "username_numeric_suffix": 0.2,
    "name_all_caps": 0.2,
    "name_short": 0.1,
    "no_avatar": 0.25,
    "bio_empty": 0.15,
    "message_repetition": 0.4,
    "high_frequency": 0.3,
    "night_activity": 0.2,
    "link_spam": 0.3,
    "media_only": 0.2,
    "language_switch": 0.15,
    "template_pattern": 0.35,
    "young_account": 0.1,
    "known_bot_id": 0.5,
    "suspicious_id_size": 0.2,
    "id_age": 0.1,
}

# ================== ХРАНИЛИЩА ==================
class UserStats:
    def __init__(self):
        self.messages = deque(maxlen=50)
        self.timestamps = deque(maxlen=50)
        self.media_count = 0
        self.link_count = 0
        self.languages = defaultdict(int)
        self.night_activity = 0
        self.first_seen = None
        self.last_seen = None
        self.has_avatar = None
        self.bio = None

violations = defaultdict(int)
user_stats: Dict[int, UserStats] = {}
bot_scores: Dict[int, float] = {}
known_bot_ids: Set[int] = set()

# ================== РАБОТА С ЧЁРНЫМ СПИСКОМ ==================
def load_blacklist():
    global known_bot_ids
    if os.path.exists(BLACKLIST_FILE):
        with open(BLACKLIST_FILE, "r") as f:
            data = json.load(f)
            known_bot_ids = set(data)
    else:
        known_bot_ids = set()

def save_blacklist():
    with open(BLACKLIST_FILE, "w") as f:
        json.dump(list(known_bot_ids), f)

def add_to_blacklist(user_id: int):
    known_bot_ids.add(user_id)
    save_blacklist()

def remove_from_blacklist(user_id: int):
    known_bot_ids.discard(user_id)
    save_blacklist()

# ================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==================
def is_username_bot(username: str) -> bool:
    return "bot" in username.lower() if username else False

def has_numeric_suffix(username: str) -> bool:
    return bool(re.search(r'\d{3,}$', username)) if username else False

def name_all_caps(first_name: str, last_name: str) -> bool:
    full = f"{first_name or ''}{last_name or ''}"
    return full.isupper() if full else False

def name_too_short(first_name: str, last_name: str) -> bool:
    return len(f"{first_name or ''}{last_name or ''}") <= 2

def is_night_time(dt: datetime) -> bool:
    return 0 <= dt.hour < 6

def extract_links(text: str) -> List[str]:
    return re.findall(r'https?://\S+', text)

def detect_template_pattern(texts: List[str]) -> bool:
    if not texts:
        return False
    counts = defaultdict(int)
    for t in texts:
        counts[t] += 1
    most_common_ratio = max(counts.values()) / len(texts)
    return most_common_ratio > 0.3

def is_suspicious_by_id(user_id: int) -> float:
    if user_id > 2_000_000_000:
        return 0.2
    return 0.0

def estimate_id_age_weight(user_id: int) -> float:
    if user_id > 2_000_000_000:
        return 0.1
    elif user_id > 1_000_000_000:
        return 0.05
    return 0.0

# ================== АСИНХРОННЫЕ ПРОВЕРКИ ==================
async def fetch_avatar_and_bio(user_id: int, bot: Bot):
    stats = user_stats.get(user_id)
    if not stats:
        return
    try:
        photos = await bot.get_user_profile_photos(user_id, limit=1)
        stats.has_avatar = photos.total_count > 0
    except:
        stats.has_avatar = False
    try:
        chat = await bot.get_chat(user_id)
        stats.bio = getattr(chat, "bio", None) if chat else None
    except:
        stats.bio = None

# ================== ОЦЕНКА БОТОВОСТИ ==================
async def calculate_bot_score(user: types.User, stats: UserStats, bot: Bot) -> float:
    score = 0.0

    if is_username_bot(user.username):
        score += FEATURE_WEIGHTS["username_bot"]
    if has_numeric_suffix(user.username):
        score += FEATURE_WEIGHTS["username_numeric_suffix"]
    if name_all_caps(user.first_name, user.last_name):
        score += FEATURE_WEIGHTS["name_all_caps"]
    if name_too_short(user.first_name, user.last_name):
        score += FEATURE_WEIGHTS["name_short"]

    if stats.has_avatar is not None and not stats.has_avatar:
        score += FEATURE_WEIGHTS["no_avatar"]
    if stats.bio is not None and not stats.bio:
        score += FEATURE_WEIGHTS["bio_empty"]

    if stats.messages:
        unique_ratio = len(set(stats.messages)) / len(stats.messages)
        if unique_ratio < 0.3:
            score += FEATURE_WEIGHTS["message_repetition"]

        if len(stats.timestamps) > 1:
            first = stats.timestamps[0]
            last = stats.timestamps[-1]
            delta = (last - first).total_seconds() / 60
            if delta > 0:
                freq = len(stats.timestamps) / delta
                if freq > 3:
                    score += FEATURE_WEIGHTS["high_frequency"]

        if stats.timestamps:
            night_ratio = stats.night_activity / len(stats.timestamps)
            if night_ratio > 0.5:
                score += FEATURE_WEIGHTS["night_activity"]

        link_ratio = stats.link_count / len(stats.messages)
        if link_ratio > 0.4:
            score += FEATURE_WEIGHTS["link_spam"]

        media_ratio = stats.media_count / len(stats.messages)
        if media_ratio > 0.7:
            score += FEATURE_WEIGHTS["media_only"]

        if len(stats.languages) > 2:
            score += FEATURE_WEIGHTS["language_switch"]

        if detect_template_pattern(list(stats.messages)):
            score += FEATURE_WEIGHTS["template_pattern"]

    if user.id in known_bot_ids:
        score += FEATURE_WEIGHTS["known_bot_id"]
    score += is_suspicious_by_id(user.id)
    score += estimate_id_age_weight(user.id)

    total_weight = sum(FEATURE_WEIGHTS.values())
    normalized = min(score / total_weight, 1.0)
    return normalized

# ================== ОБНОВЛЕНИЕ СТАТИСТИКИ ==================
def update_user_stats(user_id: int, message: types.Message):
    if user_id not in user_stats:
        user_stats[user_id] = UserStats()
        user_stats[user_id].first_seen = datetime.now()

    stats = user_stats[user_id]
    now = datetime.now()

    stats.messages.append(message.text or "")
    stats.timestamps.append(now)
    stats.last_seen = now

    if any([message.photo, message.video, message.audio, message.document]):
        stats.media_count += 1

    if message.text and extract_links(message.text):
        stats.link_count += 1

    if is_night_time(now):
        stats.night_activity += 1

    if message.text:
        if re.search(r'[а-яА-Я]', message.text):
            stats.languages['ru'] += 1
        elif re.search(r'[a-zA-Z]', message.text):
            stats.languages['en'] += 1
        else:
            stats.languages['other'] += 1

# ================== ПРОВЕРКИ ==================
def contains_bad_words(text: str) -> bool:
    text_lower = text.lower()
    for word in BAD_WORDS:
        if re.search(rf'\b{re.escape(word)}\b', text_lower):
            return True
    return False

def is_spam(user_id: int) -> bool:
    now = datetime.now()
    stats = user_stats.get(user_id)
    if not stats:
        return False
    timestamps = list(stats.timestamps)
    timestamps = [t for t in timestamps if now - t < timedelta(seconds=SPAM_INTERVAL)]
    stats.timestamps = deque(timestamps, maxlen=50)
    stats.timestamps.append(now)
    return len(timestamps) > SPAM_LIMIT

# ================== БЛОКИРОВКА ==================
async def ban_user(chat_id: int, user_id: int, reason: str, bot: Bot):
    permissions = ChatPermissions(
        can_send_messages=False,
        can_send_media=False,
        can_send_other_messages=False,
        can_add_web_page_previews=False,
        can_send_polls=False,
        can_invite_users=False,
        can_change_info=False,
        can_pin_messages=False
    )
    await bot.restrict_chat_member(chat_id, user_id, permissions)
    await bot.send_message(chat_id, f"🚫 Пользователь {user_id} заблокирован. Причина: {reason}")
    add_to_blacklist(user_id)

# ================== УВЕДОМЛЕНИЕ АДМИНИСТРАТОРОВ ==================
async def notify_manual_check(chat_id: int, user: types.User, score: float, bot: Bot):
    admins = await bot.get_chat_administrators(chat_id)
    for admin in admins:
        if not admin.user.is_bot:
            await bot.send_message(
                admin.user.id,
                f"⚠️ Требуется ручная проверка пользователя {user.full_name} (id: `{user.id}`)\n"
                f"Уверенность: {score:.2f}\n"
                f"Команды: `/verify {user.id}` – человек, `/ban {user.id}` – бан",
                parse_mode="Markdown"
            )

# ================== ОСНОВНАЯ ЛОГИКА МОДЕРАЦИИ ==================
async def analyze_and_moderate(message: types.Message, bot: Bot):
    user = message.from_user
    chat_id = message.chat.id
    text = message.text or ""

    if not text:
        return

    update_user_stats(user.id, message)

    if contains_bad_words(text):
        violations[user.id] += 1
        current_warns = violations[user.id]
        if current_warns >= WARN_LIMIT:
            await ban_user(chat_id, user.id, f"нарушения ({WARN_LIMIT})", bot)
            del violations[user.id]
        else:
            await message.reply(f"⚠️ Предупреждение {current_warns}/{WARN_LIMIT}. Не используйте запрещённые слова!")
        return

    if is_spam(user.id):
        violations[user.id] += 1
        current_warns = violations[user.id]
        if current_warns >= WARN_LIMIT:
            await ban_user(chat_id, user.id, "спам", bot)
            del violations[user.id]
        else:
            await message.reply(f"⚠️ Предупреждение {current_warns}/{WARN_LIMIT}. Слишком много сообщений!")
        return

    stats = user_stats.get(user.id)
    if stats and len(stats.timestamps) >= 5:
        if stats.has_avatar is None:
            asyncio.create_task(fetch_avatar_and_bio(user.id, bot))

        bot_score = await calculate_bot_score(user, stats, bot)
        bot_scores[user.id] = bot_score

        if bot_score >= BOT_BAN_THRESHOLD:
            await ban_user(chat_id, user.id, f"бот (уверенность {bot_score:.2f})", bot)
            return
        elif bot_score >= MANUAL_VERIFICATION_NEEDED:
            last_notify_key = f"notify_{user.id}"
            if not hasattr(analyze_and_moderate, last_notify_key) or \
               (datetime.now() - getattr(analyze_and_moderate, last_notify_key)).seconds > 300:
                await notify_manual_check(chat_id, user, bot_score, bot)
                setattr(analyze_and_moderate, last_notify_key, datetime.now())

# ================== КОМАНДЫ ==================
async def cmd_start(message: types.Message):
    await message.answer("🤖 Бот-модератор запущен. Администраторы могут использовать /info, /id, /verify, /ban, /addbot, /removebot")

async def cmd_info(message: types.Message):
    info = (
        f"📊 Статистика\n"
        f"• Запрещённых слов: {len(BAD_WORDS)}\n"
        f"• Лимит предупреждений: {WARN_LIMIT}\n"
        f"• Спам-фильтр: {SPAM_LIMIT} за {SPAM_INTERVAL} сек\n"
        f"• Порог бана бота: {BOT_BAN_THRESHOLD}\n"
        f"• ID в чёрном списке: {len(known_bot_ids)}\n"
        f"• Активных пользователей: {len(user_stats)}"
    )
    await message.answer(info)

async def cmd_id(message: types.Message):
    if message.reply_to_message:
        user = message.reply_to_message.from_user
        await message.answer(f"🆔 ID пользователя {user.full_name}: `{user.id}`", parse_mode="Markdown")
    else:
        await message.answer(f"🆔 Ваш ID: `{message.from_user.id}`", parse_mode="Markdown")

async def cmd_verify(message: types.Message, bot: Bot):
    member = await bot.get_chat_member(message.chat.id, message.from_user.id)
    if member.status not in ('creator', 'administrator'):
        return
    args = message.get_args()
    if args and args.isdigit():
        user_id = int(args)
        if user_id in known_bot_ids:
            remove_from_blacklist(user_id)
        await message.reply(f"✅ Пользователь {user_id} помечен как человек.")
    else:
        await message.reply("Использование: /verify <user_id>")

async def cmd_ban(message: types.Message, bot: Bot):
    member = await bot.get_chat_member(message.chat.id, message.from_user.id)
    if member.status not in ('creator', 'administrator'):
        return
    args = message.get_args()
    if args and args.isdigit():
        user_id = int(args)
        await ban_user(message.chat.id, user_id, "ручная блокировка", bot)
        await message.reply(f"🚫 Пользователь {user_id} заблокирован.")
    else:
        await message.reply("Использование: /ban <user_id>")

async def cmd_addbot(message: types.Message, bot: Bot):
    member = await bot.get_chat_member(message.chat.id, message.from_user.id)
    if member.status not in ('creator', 'administrator'):
        return
    args = message.get_args()
    if args and args.isdigit():
        user_id = int(args)
        add_to_blacklist(user_id)
        await message.reply(f"🤖 ID {user_id} добавлен в чёрный список ботов.")
    else:
        await message.reply("Использование: /addbot <user_id>")

async def cmd_removebot(message: types.Message, bot: Bot):
    member = await bot.get_chat_member(message.chat.id, message.from_user.id)
    if member.status not in ('creator', 'administrator'):
        return
    args = message.get_args()
    if args and args.isdigit():
        user_id = int(args)
        remove_from_blacklist(user_id)
        await message.reply(f"✅ ID {user_id} удалён из чёрного списка.")
    else:
        await message.reply("Использование: /removebot <user_id>")

# ================== РЕГИСТРАЦИЯ ХЕНДЛЕРОВ ==================
def register_handlers(dp: Dispatcher):
    dp.register_message_handler(cmd_start, commands="start")
    dp.register_message_handler(cmd_info, commands="info")
    dp.register_message_handler(cmd_id, commands="id")
    dp.register_message_handler(cmd_verify, commands="verify")
    dp.register_message_handler(cmd_ban, commands="ban")
    dp.register_message_handler(cmd_addbot, commands="addbot")
    dp.register_message_handler(cmd_removebot, commands="removebot")
    dp.register_message_handler(analyze_and_moderate, lambda msg: True)

# ================== ЗАПУСК ==================
async def on_startup(dp: Dispatcher):
    load_blacklist()
    await dp.bot.set_my_commands([
        types.BotCommand("start", "Начать"),
        types.BotCommand("info", "Статистика"),
        types.BotCommand("id", "Узнать ID"),
        types.BotCommand("verify", "Подтвердить человека (админ)"),
        types.BotCommand("ban", "Заблокировать (админ)"),
        types.BotCommand("addbot", "Добавить ID в чёрный список"),
        types.BotCommand("removebot", "Удалить ID из чёрного списка"),
    ])

if __name__ == "__main__":
    bot = Bot()
    dp = Dispatcher(bot)
    register_handlers(dp)
    executor.start_polling(dp, on_startup=on_startup, skip_updates=True)
