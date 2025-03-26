from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class RAGResponse(BaseModel):
    """Модель для структурированного ответа RAG-системы."""
    статус: str = Field(description="Статус ответа: 'успех' или 'ошибка'")
    релевантность: bool = Field(description="Релевантность запроса теме")
    краткий_ответ: Optional[str] = Field(description="Краткий ответ на вопрос (1-2 предложения)")
    основные_факты: Optional[List[str]] = Field(description="Список основных фактов из источников")
    источники: Optional[List[Dict[str, str]]] = Field(description="Список использованных источников с датами")
    ошибка: Optional[str] = Field(description="Описание ошибки, если статус 'ошибка'") 