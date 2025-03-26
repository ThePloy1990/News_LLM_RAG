# Crypto News Aggregator с RAG

Система для сбора и анализа новостей о криптовалютах из Twitter с использованием Retrieval-Augmented Generation (RAG).

## Требования

- Python 3.7+
- Учетная запись разработчика Twitter (для получения Bearer Token)

## Установка

1. Клонируйте репозиторий:
```bash
git clone <url-репозитория>
cd crypto_news_aggregator
```

2. Создайте виртуальное окружение и установите зависимости:
```bash
python -m venv venv
# Для Windows:
venv\Scripts\activate 
# Для Linux/Mac:
# source venv/bin/activate
pip install -r requirements.txt
```

3. Создайте файл `.env` в корневой директории проекта и добавьте ваш Twitter Bearer Token:
```
TWITTER_BEARER_TOKEN=ваш_токен_здесь
```

4. (Опционально) Для использования локальной языковой модели вместо заглушки:
   - Скачайте модель Phi-3-mini-4k-instruct.Q4_K_M.gguf и поместите ее в директорию `models/`
   - Установите llama-cpp-python: `pip install llama-cpp-python`
   - Обновите код в `src/rag/phi_llm.py`, чтобы использовать реальную модель

## Настройка

Редактируйте файл `src/fetch_twitter.py` для указания интересующих вас Twitter-аккаунтов:

```python
twitter_users = [
    "VitalikButerin",    # Виталик Бутерин
    "cz_binance",        # Чанпэн Чжао (Binance)
    "saylor",            # Майкл Сейлор
    "SBF_FTX",           # Сэм Бэнкман-Фрид
    "elonmusk"           # Илон Маск
    # Добавьте других интересующих вас пользователей
]
```

## Использование

### Сбор данных из Twitter

```bash
python src/app.py collect-data
```

### Создание индекса для RAG

```bash
python src/app.py update-index
```

### Поиск релевантных документов

```bash
python src/app.py search --query "Ваш поисковый запрос"
```

### Анализ в режиме вопрос-ответ (QA)

```bash
python src/app.py analyze --mode qa --query "Ваш вопрос"
```

### Сравнение мнений разных авторов

```bash
python src/app.py analyze --mode compare --query "Тема для сравнения"
```

## Структура проекта

- `src/` - исходный код проекта
  - `app.py` - основной скрипт с CLI-интерфейсом
  - `fetch_twitter.py` - сбор данных из Twitter
  - `prepare_data.py` - подготовка и объединение данных
  - `rag/` - модули для RAG-системы
    - `rag_pipeline.py` - основной пайплайн RAG
    - `build_index.py` - создание векторного индекса
    - `phi_llm.py` - интерфейс к языковой модели
- `data/` - директория для хранения данных
- `faiss_index/` - директория для хранения векторного индекса
- `models/` - директория для хранения языковых моделей

## Примечания

- Текущая реализация использует заглушку вместо реальной языковой модели. Для полноценной работы рекомендуется заменить заглушку на Microsoft Phi-3, DeepSeek или другую совместимую модель.
- Для использования Phi-3 mini в инференсе необходимо около 8 ГБ ОЗУ и совместимый с CUDA GPU.
- Если у вас нет доступа к мощному GPU, рассмотрите возможность использования API OpenAI или другого облачного сервиса для генерации ответов. 