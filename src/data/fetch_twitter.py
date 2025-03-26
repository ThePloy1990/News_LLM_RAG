import os
import json
from datetime import datetime
from typing import List
from dotenv import load_dotenv
import tweepy
import time
import random
import argparse

from .models import NewsItem

# Загрузка переменных окружения
load_dotenv()

# Конфигурация
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Список пользователей для мониторинга
# Инфлюенсеры и основатели
INFLUENCERS = [
    "VitalikButerin",    # Виталик Бутерин, создатель Ethereum
    "cz_binance",        # Чанпэн Чжао, бывший CEO Binance (лысый с банана)
    "SatoshiLite",       # Чарли Ли, создатель Litecoin
    "gavofyork",         # Гэвин Вуд, основатель Polkadot (када 50 баксов брат)
    "justin_sun",        # Джастин Сан, основатель TRON (jail graduated 2025)
    "IOHK_Charles",      # Чарльз Хоскинсон, создатель Cardano (дешевая памп девка трампа)
    "AriannaSimpson",    # Арианна Симпсон, партнер a16z Crypto (...)
    "brian_armstrong",   # Брайан Армстронг, CEO Coinbase (RAHHHHH WHATAFUCK IS A KILOMETERRRRRRRR RAAHHHHH)
    "cdixon",            # Крис Диксон, партнер a16z (...)
    "bhorowitz",         # Бен Хоровитц, партнер a16z (...)
    "jessepollak",       # Джесс Поллак, руководитель Base в Coinbase (ai мета кормит)
]

# Трейдеры, аналитики и инвесторы
TRADERS = [
    "AiXbT",             # AiXbT LLM-трейдер (нейронка с пасом за 600к долеров)
    "rektcapital",       # Rekt Capital, технический аналитик (хз лудос какой-то чето там чертит на графиках, в твиттер скидывает)
    "APompliano",        # Энтони Помплиано, инвестор хзхзхз
    "CryptoMichNL",      # Микаэль ван де Поппе, трейдер и аналитик (хз кто такой)
    "PeterLBrandt",      # Питер Брандт, опытный трейдер (хз кто такой)
    "woonomic",          # Вилли Ву, криптоаналитик on-chain данных (хз кто такой)
    "ToneVays",          # Тон Вэйс, трейдер и аналитик (хз кто такой)
    "DanHeld",           # Дэн Хелд, биткоин-максималист (лудик 2)
    "100trillionUSD",    # PlanB, создатель модели S2F
    "RaoulGMI",          # Рауль Пал, CEO Real Vision
    "novogratz",         # Майк Новограц, CEO Galaxy Digital
    "DTAPCAP",           # Дэн Тапиеро, макро-инвестор
    "Bitboy_Crypto",     # Бен Армстронг, криптовалютный ютубер
    "scottmelker",       # Скотт Мелкер, трейдер и подкастер
    "Crypto_Birb",       # Биткоин-аналитик и чарт-трейдер
]

# Киты и дегены
WHALES = [
    "SirlulzAlot",       # Известный NFT-трейдер
    "loomdart",          # Анонимный трейдер и инвестор
    "DegenSpartan",      # Анонимный деген-трейдер 
    "CryptoCobain",      # Известный криптоинвестор
    "girlgone_crypto",   # Лия Хок, криптоинфлюенсер
    "Arthur_0x",         # Артур Чен, сооснователь DeFiance Capital
    "zhusu",             # Су Чжу, сооснователь Three Arrows Capital
    "crypto_chica",      # Криптоинфлюенсер и трейдер
    "0xGhostchain",      # Анонимный DeFi-эксперт
    "0xfoobar",          # Анонимный DeFi-разработчик 
]

# Безопасность и расследования
SECURITY = [
    "zachxbt",           # ZachXBT, крипто-детектив, разоблачающий скамы
    "NFTherder",         # OKHotShot, аналитик NFT-рынка
    "0xQuit",            # Специалист по безопасности в Web3
    "officer_cia",       # Анонимный аналитик и исследователь
    "tayvano_",          # Тейлор Монахан, основатель MyCrypto, эксперт безопасности
]

# Новостные источники и инструменты
NEWS = [
    "whale_alert",       # Отслеживает крупные транзакции в блокчейне
    "BitcoinMagazine",   # Ведущее издание о биткоине
    "Cointelegraph",     # Криптовалютное новостное издание
    "coindesk",          # Криптовалютное новостное издание
    "glassnode",         # Аналитическая платформа on-chain данных
    "santimentfeed",     # Аналитическая платформа рыночных данных
    "MessariCrypto",     # Исследовательская платформа
    "cryptoquant_com",   # Аналитическая платформа данных
    "DefiPulse",         # Мониторинг DeFi проектов
    "WuBlockchain",      # Колин Ву, криптожурналист с фокусом на Азию
    "tier10k",           # Инсайдерская рассылка о криптомире
    "TheBlock__",        # Исследовательская и новостная платформа
]

# NFT и Web3 специалисты
NFT = [
    "beeple",            # Майк Винкельманн, NFT-артист
    "jimykim",           # Джими Ким, NFT-коллекционер
    "punk6529",          # Анонимный NFT-коллекционер и мыслитель
    "pranksy",           # Известный NFT-коллекционер и инвестор
    "CozomoMedici",      # Анонимный NFT-коллекционер
    "DCLBlogger",        # Эксперт по метавселенным
    "j1mmyeth",          # Jimmy.eth, NFT-инвестор
]

# Объединяем все группы в единый список
TWITTER_USERS = INFLUENCERS + TRADERS + WHALES + SECURITY + NEWS + NFT

class TwitterFetcher:
    def __init__(self):
        self.client = tweepy.Client(bearer_token=BEARER_TOKEN)
        self.processed_ids = set()

    def _load_processed_ids(self):
        """Загрузка уже обработанных ID твитов."""
        try:
            with open("data/processed_twitter_ids.json", "r") as f:
                self.processed_ids = set(json.load(f))
        except FileNotFoundError:
            self.processed_ids = set()

    def _save_processed_ids(self):
        """Сохранение обработанных ID твитов."""
        os.makedirs("data", exist_ok=True)
        with open("data/processed_twitter_ids.json", "w") as f:
            json.dump(list(self.processed_ids), f)

    def fetch_user_tweets(self, username: str, max_results: int = 100) -> List[NewsItem]:
        """Получение твитов пользователя."""
        messages = []
        try:
            # Получаем ID пользователя
            user = self.client.get_user(username=username)
            if not user.data:
                print(f"Пользователь {username} не найден")
                return messages

            user_id = user.data.id
            
            # Определяем категорию пользователя
            category = None
            if username in INFLUENCERS:
                category = "influencer"
            elif username in TRADERS:
                category = "trader"
            elif username in WHALES:
                category = "whale"
            elif username in SECURITY:
                category = "security"
            elif username in NEWS:
                category = "news"
            elif username in NFT:
                category = "nft"

            # Получаем твиты
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=max_results,
                tweet_fields=["created_at", "public_metrics"]
            )

            if not tweets.data:
                print(f"Нет доступных твитов от пользователя {username}")
                return messages

            for tweet in tweets.data:
                if str(tweet.id) in self.processed_ids:
                    continue

                news_item = NewsItem(
                    source="twitter",
                    author=username,
                    text=tweet.text,
                    date=tweet.created_at,
                    id=str(tweet.id),
                    url=f"https://twitter.com/{username}/status/{tweet.id}",
                    metadata={
                        "retweet_count": tweet.public_metrics["retweet_count"],
                        "reply_count": tweet.public_metrics["reply_count"],
                        "like_count": tweet.public_metrics["like_count"],
                        "quote_count": tweet.public_metrics["quote_count"]
                    },
                    category=category
                )
                messages.append(news_item)
                self.processed_ids.add(str(tweet.id))

        except tweepy.TweepyException as e:
            # Обрабатываем специфические ошибки API
            if "429" in str(e):
                print(f"Превышен лимит запросов API для пользователя {username}: {e}")
                # Если превышен лимит запросов, делаем более длинную паузу
                time.sleep(random.uniform(30, 60))
            else:
                print(f"Ошибка API при получении твитов пользователя {username}: {e}")
        except Exception as e:
            print(f"Непредвиденная ошибка при получении твитов пользователя {username}: {e}")

        return messages

    def fetch_all_users(self, max_tweets: int = 10) -> List[NewsItem]:
        """Получение твитов от всех пользователей."""
        all_messages = []
        self._load_processed_ids()

        # Разбиваем всех пользователей на небольшие группы по 5 пользователей
        batch_size = 5
        user_batches = [TWITTER_USERS[i:i + batch_size] for i in range(0, len(TWITTER_USERS), batch_size)]
        
        print(f"Пользователи разделены на {len(user_batches)} групп по {batch_size} пользователей")
        
        for batch_num, user_batch in enumerate(user_batches):
            print(f"\nОбработка группы {batch_num + 1}/{len(user_batches)}...")
            
            for user in user_batch:
                try:
                    messages = self.fetch_user_tweets(user, max_results=max_tweets)
                    all_messages.extend(messages)
                    # Добавляем случайную задержку между запросами (от 2 до 5 секунд)
                    delay = random.uniform(2, 5)
                    print(f"Задержка {delay:.1f} сек перед следующим запросом...")
                    time.sleep(delay)
                except Exception as e:
                    print(f"Ошибка при получении твитов пользователя {user}: {e}")
            
            # Если это не последняя группа, делаем паузу между группами
            if batch_num < len(user_batches) - 1:
                # Добавляем длительную задержку между группами (от 15 до 30 секунд)
                group_delay = random.uniform(15, 30)
                print(f"\nПауза между группами: {group_delay:.1f} сек...")
                time.sleep(group_delay)

        self._save_processed_ids()
        return all_messages

    def fetch_limited_users(self, max_users: int = 10, max_tweets: int = 10) -> List[NewsItem]:
        """Получение твитов от ограниченного числа пользователей."""
        all_messages = []
        self._load_processed_ids()
        
        # Выбираем случайных пользователей из каждой категории
        limited_users = []
        
        # Функция для получения случайных пользователей из категории
        def get_random_from_category(category, num):
            return random.sample(category, min(len(category), num))
        
        # Получаем по несколько случайных пользователей из каждой категории
        limited_users.extend(get_random_from_category(INFLUENCERS, max_users // 6))
        limited_users.extend(get_random_from_category(TRADERS, max_users // 6))
        limited_users.extend(get_random_from_category(WHALES, max_users // 6))
        limited_users.extend(get_random_from_category(SECURITY, max_users // 6))
        limited_users.extend(get_random_from_category(NEWS, max_users // 6))
        limited_users.extend(get_random_from_category(NFT, max_users // 6))
        
        # Перемешиваем список
        random.shuffle(limited_users)
        
        # Ограничиваем список до максимального числа пользователей
        limited_users = limited_users[:max_users]
        
        print(f"Выбрано {len(limited_users)} пользователей для обработки:")
        for user in limited_users:
            print(f"  - {user}")
        
        # Разбиваем пользователей на небольшие группы
        batch_size = 3
        user_batches = [limited_users[i:i + batch_size] for i in range(0, len(limited_users), batch_size)]
        
        for batch_num, user_batch in enumerate(user_batches):
            print(f"\nОбработка группы {batch_num + 1}/{len(user_batches)}...")
            
            for user in user_batch:
                try:
                    messages = self.fetch_user_tweets(user, max_results=max_tweets)
                    all_messages.extend(messages)
                    # Добавляем случайную задержку между запросами
                    delay = random.uniform(2, 5)
                    print(f"Задержка {delay:.1f} сек перед следующим запросом...")
                    time.sleep(delay)
                except Exception as e:
                    print(f"Ошибка при получении твитов пользователя {user}: {e}")
            
            # Если это не последняя группа, делаем паузу между группами
            if batch_num < len(user_batches) - 1:
                group_delay = random.uniform(15, 30)
                print(f"\nПауза между группами: {group_delay:.1f} сек...")
                time.sleep(group_delay)

        self._save_processed_ids()
        return all_messages

    def save_messages(self, messages: List[NewsItem]):
        """Сохранение сообщений в JSON файл."""
        os.makedirs("data", exist_ok=True)

        # Загрузка существующих сообщений
        existing_messages = []
        try:
            with open("data/twitter_dump.json", "r", encoding="utf-8") as f:
                existing_messages = json.load(f)
        except FileNotFoundError:
            pass

        # Добавление новых сообщений
        new_messages = [vars(msg) for msg in messages]
        all_messages = existing_messages + new_messages

        # Сохранение обновленного списка
        with open("data/twitter_dump.json", "w", encoding="utf-8") as f:
            json.dump(all_messages, f, ensure_ascii=False, indent=2, default=str)

    def print_category_stats(self):
        """Вывод статистики по категориям пользователей."""
        print("\n=== Статистика по категориям пользователей ===")
        categories = {
            "influencer": len(INFLUENCERS),
            "trader": len(TRADERS),
            "whale": len(WHALES),
            "security": len(SECURITY),
            "news": len(NEWS),
            "nft": len(NFT)
        }
        
        total = len(TWITTER_USERS)
        print(f"Всего пользователей: {total}")
        
        for category, count in categories.items():
            percentage = (count / total) * 100
            print(f"{category.capitalize()}: {count} ({percentage:.1f}%)")


def main():
    """Основная функция для запуска сбора данных."""
    parser = argparse.ArgumentParser(description="Сбор данных из Twitter")
    parser.add_argument("--limit", type=int, default=0, 
                       help="Ограничить число пользователей для обработки (0 - обработка всех пользователей)")
    parser.add_argument("--max-tweets", type=int, default=10, 
                       help="Максимальное количество твитов от одного пользователя (1-100)")
    args = parser.parse_args()
    
    # Проверяем аргументы
    if args.max_tweets < 1 or args.max_tweets > 100:
        print("Ошибка: количество твитов должно быть от 1 до 100")
        return
    
    fetcher = TwitterFetcher()
    fetcher.print_category_stats()
    
    print("\nВНИМАНИЕ: Twitter API имеет строгие ограничения на количество запросов.")
    print("Для обхода этих ограничений пользователи обрабатываются группами с задержками.")
    print("Это займет некоторое время, пожалуйста, будьте терпеливы.\n")
    
    if args.limit > 0:
        print(f"Режим ограниченного сбора: обработка не более {args.limit} пользователей")
        print(f"Максимум {args.max_tweets} твитов от каждого пользователя")
        messages = fetcher.fetch_limited_users(args.limit, max_tweets=args.max_tweets)
    else:
        print("Полный режим сбора: обработка всех пользователей")
        print(f"Максимум {args.max_tweets} твитов от каждого пользователя")
        messages = fetcher.fetch_all_users(max_tweets=args.max_tweets)
    
    fetcher.save_messages(messages)
    print(f"Собрано {len(messages)} новых твитов")


if __name__ == "__main__":
    main() 