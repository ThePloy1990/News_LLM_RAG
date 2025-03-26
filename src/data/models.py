from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class NewsItem:
    """Модель для хранения новостного поста."""
    source: str  # 'twitter' или 'telegram'
    author: str
    text: str
    date: datetime
    url: Optional[str] = None
    id: Optional[str] = None  # Уникальный идентификатор поста
    metadata: Optional[dict] = None  # Дополнительные метаданные
    category: Optional[str] = None  # Категория автора (influencer, trader, whale, security, news, nft) 