import os
import json
from datetime import datetime
from typing import List, Dict
import re

from .models import NewsItem


def load_json_file(filepath: str) -> List[Dict]:
    """Загрузка данных из JSON файла."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Файл {filepath} не найден")
        return []


def clean_text(text: str) -> str:
    """Очистка текста от лишних элементов."""
    # Удаление URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Удаление хештегов
    text = re.sub(r'#\w+', '', text)
    
    # Удаление упоминаний
    text = re.sub(r'@\w+', '', text)
    
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def prepare_data() -> List[Dict]:
    """Подготовка данных из Twitter."""
    # Загрузка данных
    twitter_data = load_json_file("data/twitter_dump.json")
    
    # Очистка и фильтрация
    processed_data = []
    seen_texts = set()  # Для предотвращения дубликатов
    
    for item in twitter_data:
        # Пропускаем слишком короткие сообщения
        if len(item["text"]) < 10:
            continue
            
        # Очищаем текст
        cleaned_text = clean_text(item["text"])
        
        # Пропускаем пустые сообщения после очистки
        if not cleaned_text:
            continue
            
        # Пропускаем дубликаты
        if cleaned_text in seen_texts:
            continue
            
        # Обновляем текст
        item["text"] = cleaned_text
        
        # Преобразуем строковую дату в datetime
        if isinstance(item["date"], str):
            try:
                item["date"] = datetime.fromisoformat(item["date"].replace('Z', '+00:00'))
            except ValueError:
                continue
        
        # Проверяем наличие поля category
        if "category" not in item:
            item["category"] = None
        
        processed_data.append(item)
        seen_texts.add(cleaned_text)
    
    # Сортировка по дате (новые первыми)
    processed_data.sort(key=lambda x: x["date"], reverse=True)
    
    return processed_data


def save_processed_data(data: List[Dict]):
    """Сохранение обработанных данных."""
    os.makedirs("data", exist_ok=True)
    
    with open("data/all_posts.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    # Также сохраняем данные, разделенные по категориям
    by_category = {}
    for item in data:
        category = item.get("category") or "uncategorized"
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(item)
    
    # Создаем директорию для категорий
    os.makedirs("data/categories", exist_ok=True)
    
    # Сохраняем файлы по категориям
    for category, items in by_category.items():
        with open(f"data/categories/{category}_posts.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2, default=str)


def main():
    """Основная функция для запуска подготовки данных."""
    print("Начинаем подготовку данных...")
    
    processed_data = prepare_data()
    save_processed_data(processed_data)
    
    # Выводим статистику по категориям
    categories = {}
    for item in processed_data:
        category = item.get("category") or "uncategorized"
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    
    print(f"Обработано {len(processed_data)} сообщений")
    print("\n=== Статистика по категориям ===")
    for category, count in categories.items():
        percentage = (count / len(processed_data)) * 100
        print(f"{category.capitalize()}: {count} сообщений ({percentage:.1f}%)")
    
    print("\nДанные сохранены в data/all_posts.json и по категориям в data/categories/")


if __name__ == "__main__":
    main()