import os
import json
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.functions.messages import GetHistoryRequest

from .models import NewsItem

# Загрузка переменных окружения
load_dotenv()

# Конфигурация
API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION_FILE = "telegram_session"

# Список каналов для мониторинга
TELEGRAM_CHANNELS = [
    "example_channel1",  # Замените на реальные каналы
    "example_channel2"
]

class TelegramFetcher:
    def __init__(self):
        self.client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
        self.processed_ids = set()

    async def _load_processed_ids(self):
        """Загрузка уже обработанных ID сообщений."""
        try:
            with open("data/processed_telegram_ids.json", "r") as f:
                self.processed_ids = set(json.load(f))
        except FileNotFoundError:
            self.processed_ids = set()

    async def _save_processed_ids(self):
        """Сохранение обработанных ID сообщений."""
        os.makedirs("data", exist_ok=True)
        with open("data/processed_telegram_ids.json", "w") as f:
            json.dump(list(self.processed_ids), f)

    async def fetch_messages(self, channel: str, limit: int = 100) -> List[NewsItem]:
        """Получение сообщений из канала."""
        messages = []
        try:
            history = await self.client(GetHistoryRequest(
                peer=channel,
                offset_id=0,
                offset_date=None,
                add_offset=0,
                limit=limit,
                max_id=0,
                min_id=0,
                hash=0
            ))
            
            for msg in history.messages:
                if str(msg.id) in self.processed_ids:
                    continue
                    
                if not msg.message:  # Пропускаем сообщения без текста
                    continue
                    
                news_item = NewsItem(
                    source="telegram",
                    author=channel,
                    text=msg.message,
                    date=msg.date,
                    id=str(msg.id),
                    metadata={
                        "views": getattr(msg, "views", None),
                        "forwards": getattr(msg, "forwards", None),
                        "replies": getattr(msg, "replies", None)
                    }
                )
                messages.append(news_item)
                self.processed_ids.add(str(msg.id))
                
        except Exception as e:
            print(f"Ошибка при получении сообщений из канала {channel}: {e}")
            
        return messages

    async def fetch_all_channels(self) -> List[NewsItem]:
        """Получение сообщений из всех каналов."""
        all_messages = []
        await self._load_processed_ids()
        
        for channel in TELEGRAM_CHANNELS:
            messages = await self.fetch_messages(channel)
            all_messages.extend(messages)
            
        await self._save_processed_ids()
        return all_messages

    async def save_messages(self, messages: List[NewsItem]):
        """Сохранение сообщений в JSON файл."""
        os.makedirs("data", exist_ok=True)
        
        # Загрузка существующих сообщений
        existing_messages = []
        try:
            with open("data/telegram_dump.json", "r", encoding="utf-8") as f:
                existing_messages = json.load(f)
        except FileNotFoundError:
            pass
            
        # Добавление новых сообщений
        new_messages = [vars(msg) for msg in messages]
        all_messages = existing_messages + new_messages
        
        # Сохранение обновленного списка
        with open("data/telegram_dump.json", "w", encoding="utf-8") as f:
            json.dump(all_messages, f, ensure_ascii=False, indent=2, default=str)

async def main():
    """Основная функция для запуска сбора данных."""
    fetcher = TelegramFetcher()
    
    try:
        await fetcher.client.start()
        
        if not await fetcher.client.is_user_authorized():
            try:
                await fetcher.client.sign_in(phone=input("Введите номер телефона: "))
            except SessionPasswordNeededError:
                await fetcher.client.sign_in(password=input("Введите пароль 2FA: "))
        
        messages = await fetcher.fetch_all_channels()
        await fetcher.save_messages(messages)
        print(f"Собрано {len(messages)} новых сообщений")
        
    finally:
        await fetcher.client.disconnect()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 