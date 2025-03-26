#!/usr/bin/env python
"""
Модуль для измерения производительности различных моделей в RAG-системе.
Измеряет скорость инференса, использование памяти и GPU.
"""
import os
import time
import json
import argparse
import gc
import numpy as np
import torch
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from src.rag.llm_factory import LLMFactory
from src.rag.rag_pipeline import RAGPipeline

# Загружаем переменные окружения
load_dotenv()

# Тестовые запросы разной сложности
BENCHMARK_QUERIES = [
    {
        "id": "simple",
        "text": "Что такое биткоин?",
        "description": "Простой запрос для определения основного понятия",
        "complexity": "low"
    },
    {
        "id": "medium",
        "text": "Какие основные преимущества и недостатки proof-of-stake по сравнению с proof-of-work?",
        "description": "Запрос средней сложности, требующий технического сравнения",
        "complexity": "medium"
    },
    {
        "id": "complex", 
        "text": "Какие основные факторы повлияли на изменение курса Ethereum в 2023 году и какие перспективы развития прогнозируются экспертами?",
        "description": "Сложный запрос, требующий анализа многих факторов",
        "complexity": "high"
    },
    {
        "id": "analytical",
        "text": "Сравните основные подходы к масштабированию блокчейна (Layer 2, шардинг, sidechains) и их эффективность для решения проблемы производительности.",
        "description": "Аналитический запрос, требующий систематизации информации",
        "complexity": "high"
    }
]

def measure_gpu_stats():
    """Измеряет статистику использования GPU."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    # Получаем информацию о GPU
    gpu_info = {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
        "memory_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        "max_memory_mb": torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    }
    
    return gpu_info

def run_model_benchmark(model_id: str, mode: str = "qa", queries: List[Dict] = None,
                         warmup: bool = True, verbose: bool = True, use_gpu: bool = True) -> Dict[str, Any]:
    """
    Запускает бенчмарк для указанной модели.
    
    Args:
        model_id: Идентификатор модели для тестирования
        mode: Режим работы ('qa' или 'compare')
        queries: Список запросов для тестирования
        warmup: Выполнить разогрев модели перед тестированием
        verbose: Подробный вывод результатов
        use_gpu: Использовать GPU если доступен
        
    Returns:
        Словарь с результатами бенчмарка
    """
    if queries is None:
        queries = BENCHMARK_QUERIES
    
    # Создаем пайплайн с указанной моделью
    if verbose:
        print(f"\n=== Бенчмарк модели: {model_id} ===")
    
    # Очищаем кэш CUDA перед измерениями
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Инициализируем модель
    start_init = time.time()
    pipeline = RAGPipeline(
        model_id=model_id,
        use_gpu=use_gpu
    )
    init_time = time.time() - start_init
    
    if verbose:
        print(f"Время инициализации: {init_time:.2f} сек")
    
    # Статистика GPU после инициализации
    gpu_after_init = measure_gpu_stats()
    
    # Разогрев модели
    if warmup and queries:
        if verbose:
            print("Разогрев модели...")
        warmup_query = "Что такое криптовалюта?"
        if mode == "qa":
            pipeline.generate_qa_response(warmup_query)
        else:
            pipeline.compare_opinions(warmup_query)
    
    # Результаты по запросам
    query_results = []
    
    # Измерение для каждого запроса
    for query in queries:
        if verbose:
            print(f"\nЗапрос [{query['id']}]: {query['text']}")
        
        # Освобождаем память перед каждым запросом
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Замеряем время выполнения запроса
        start_time = time.time()
        
        try:
            if mode == "qa":
                response = pipeline.generate_qa_response(query["text"])
            else:
                response = pipeline.compare_opinions(query["text"])
            
            success = True
            error_message = None
        except Exception as e:
            response = str(e)
            success = False
            error_message = str(e)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Получаем статистику GPU
        gpu_stats = measure_gpu_stats()
        
        # Собираем результаты по запросу
        query_result = {
            "query_id": query["id"],
            "query_text": query["text"],
            "query_complexity": query.get("complexity", "medium"),
            "success": success,
            "error": error_message,
            "time_seconds": elapsed_time,
            "tokens_per_second": None,  # Нужно доработать для получения количества токенов
            "gpu_memory_mb": gpu_stats.get("memory_allocated_mb", 0) if gpu_stats.get("available", False) else 0
        }
        
        query_results.append(query_result)
        
        if verbose:
            print(f"Время выполнения: {elapsed_time:.2f} сек")
            print(f"Результат: {'Успешно' if success else 'Ошибка: ' + error_message}")
    
    # Итоговые результаты
    results = {
        "model_id": model_id,
        "mode": mode,
        "initialization_time": init_time,
        "gpu_available": torch.cuda.is_available(),
        "gpu_info": gpu_after_init,
        "query_results": query_results,
        "summary": {
            "avg_time": np.mean([q["time_seconds"] for q in query_results]),
            "min_time": np.min([q["time_seconds"] for q in query_results]),
            "max_time": np.max([q["time_seconds"] for q in query_results]),
            "success_rate": sum(1 for q in query_results if q["success"]) / len(query_results)
        }
    }
    
    if verbose:
        print("\n=== Итоговые результаты ===")
        print(f"Средн. время: {results['summary']['avg_time']:.2f} сек")
        print(f"Мин. время: {results['summary']['min_time']:.2f} сек")
        print(f"Макс. время: {results['summary']['max_time']:.2f} сек")
        print(f"Успешных: {results['summary']['success_rate'] * 100:.1f}%")
    
    return results

def run_comparison_benchmark(model_ids: List[str], mode: str = "qa", 
                           queries: List[Dict] = None, save_results: bool = True,
                           plot_results: bool = True):
    """
    Запускает сравнение производительности нескольких моделей.
    
    Args:
        model_ids: Список идентификаторов моделей для тестирования
        mode: Режим работы ('qa' или 'compare')
        queries: Список запросов для тестирования
        save_results: Сохранять результаты в файл
        plot_results: Построить график результатов
    """
    if queries is None:
        queries = BENCHMARK_QUERIES
    
    print("\n===== Сравнительный бенчмарк моделей =====")
    print(f"Модели для тестирования: {', '.join(model_ids)}")
    print(f"Количество запросов: {len(queries)}")
    print(f"Режим: {mode}")
    
    all_results = []
    
    for model_id in model_ids:
        results = run_model_benchmark(model_id, mode, queries, verbose=True)
        all_results.append(results)
        
        # Даем системе время для освобождения ресурсов
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)
    
    # Создаем директорию для результатов
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Сохраняем результаты в файл
    if save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results/model_comparison_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Результаты сохранены в {filename}")
    
    # Строим графики сравнения
    if plot_results:
        plot_comparison_results(all_results)

def plot_comparison_results(results: List[Dict]):
    """Строит графики для сравнения результатов бенчмарка."""
    # Создаем DataFrame для удобного анализа
    rows = []
    
    for model_result in results:
        model_id = model_result["model_id"]
        
        # Базовая информация
        base_info = {
            "model_id": model_id,
            "init_time": model_result["initialization_time"],
            "avg_time": model_result["summary"]["avg_time"],
            "success_rate": model_result["summary"]["success_rate"] * 100
        }
        
        # Добавляем информацию по каждому запросу
        for query in model_result["query_results"]:
            row = base_info.copy()
            row.update({
                "query_id": query["query_id"],
                "query_complexity": query["query_complexity"],
                "query_time": query["time_seconds"],
                "success": query["success"],
                "gpu_memory_mb": query["gpu_memory_mb"]
            })
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Создаем директорию для графиков
    os.makedirs("benchmark_results/plots", exist_ok=True)
    
    # График 1: Среднее время выполнения запросов
    plt.figure(figsize=(10, 6))
    avg_by_model = df.groupby("model_id")["query_time"].mean().sort_values()
    avg_by_model.plot(kind="bar", color="skyblue")
    plt.title("Среднее время выполнения запроса по моделям")
    plt.ylabel("Время (сек)")
    plt.xlabel("Модель")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("benchmark_results/plots/avg_query_time.png")
    
    # График 2: Время выполнения по сложности запросов
    plt.figure(figsize=(12, 7))
    query_times = df.pivot_table(
        index="model_id", 
        columns="query_complexity", 
        values="query_time", 
        aggfunc="mean"
    ).sort_values(by="high" if "high" in df["query_complexity"].unique() else "medium")
    
    query_times.plot(kind="bar")
    plt.title("Время выполнения по сложности запросов")
    plt.ylabel("Время (сек)")
    plt.xlabel("Модель")
    plt.xticks(rotation=45)
    plt.legend(title="Сложность")
    plt.tight_layout()
    plt.savefig("benchmark_results/plots/query_complexity_time.png")
    
    # График 3: Успешность выполнения
    plt.figure(figsize=(10, 6))
    success_by_model = df.groupby("model_id")["success"].mean() * 100
    success_by_model.plot(kind="bar", color="lightgreen")
    plt.title("Успешность выполнения запросов (%)")
    plt.ylabel("Успешно (%)")
    plt.xlabel("Модель")
    plt.xticks(rotation=45)
    plt.ylim([0, 105])  # Добавляем немного места сверху для подписей
    plt.tight_layout()
    plt.savefig("benchmark_results/plots/success_rate.png")
    
    # График 4: Использование GPU памяти
    if "gpu_memory_mb" in df.columns and df["gpu_memory_mb"].max() > 0:
        plt.figure(figsize=(10, 6))
        gpu_by_model = df.groupby("model_id")["gpu_memory_mb"].mean().sort_values()
        gpu_by_model.plot(kind="bar", color="coral")
        plt.title("Среднее использование памяти GPU")
        plt.ylabel("Память (МБ)")
        plt.xlabel("Модель")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("benchmark_results/plots/gpu_memory_usage.png")
    
    print(f"Графики сохранены в директории benchmark_results/plots/")

def main():
    parser = argparse.ArgumentParser(description="Бенчмарк моделей для RAG-системы")
    parser.add_argument("--models", type=str, nargs="+", default=["phi", "qwen"],
                        help="Список моделей для тестирования")
    parser.add_argument("--mode", type=str, choices=["qa", "compare"], default="qa",
                        help="Режим работы (qa или compare)")
    parser.add_argument("--no-gpu", action="store_true", 
                        help="Не использовать GPU даже если доступен")
    args = parser.parse_args()
    
    print("\n===== Бенчмарк производительности моделей =====")
    print(f"Режим: {args.mode}")
    print(f"Модели для тестирования: {', '.join(args.models)}")
    print(f"Использование GPU: {'Отключено' if args.no_gpu else 'Включено, если доступен'}")
    
    run_comparison_benchmark(
        model_ids=args.models,
        mode=args.mode,
        save_results=True,
        plot_results=True
    )

if __name__ == "__main__":
    main() 