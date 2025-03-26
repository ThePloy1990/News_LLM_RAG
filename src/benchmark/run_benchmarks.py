#!/usr/bin/env python
"""
Главный скрипт для запуска всех бенчмарков и создания сводного отчета.
"""
import os
import time
import json
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.benchmark.model_performance import run_comparison_benchmark as benchmark_performance
from src.benchmark.rag_quality import compare_models_quality as benchmark_quality

# Загружаем переменные окружения
load_dotenv()

def create_report(performance_results: List[Dict], quality_results: List[Dict], report_dir: str):
    """
    Создает сводный отчет по результатам всех бенчмарков.
    
    Args:
        performance_results: Результаты бенчмарка производительности
        quality_results: Результаты бенчмарка качества
        report_dir: Директория для сохранения отчета
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Собираем все данные в один DataFrame
    df_performance = pd.DataFrame(performance_results)
    df_quality = pd.DataFrame(quality_results)
    
    # Объединяем данные
    df = pd.merge(
        df_performance,
        df_quality,
        on="model_id",
        suffixes=["_perf", "_qual"]
    )
    
    # Создаем сводную таблицу
    summary_table = pd.DataFrame({
        "Модель": df["model_id"],
        "Среднее время ответа (сек)": df["summary_perf"].apply(lambda x: x["avg_time"]),
        "Успешных запросов (%)": df["summary_perf"].apply(lambda x: x["success_rate"] * 100),
        "Покрытие ключевых слов (%)": df["avg_keyword_coverage_percent"],
        "Частота цитирования (%)": df["citation_rate_percent"],
        "Средн. длина ответа (слов)": df["avg_word_count"]
    })
    
    # Создаем директорию для отчета
    os.makedirs(report_dir, exist_ok=True)
    
    # Сохраняем сводную таблицу в CSV
    summary_table.to_csv(f"{report_dir}/summary_table.csv", index=False)
    
    # Создаем HTML отчет
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Отчет о бенчмаркинге RAG-системы</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .chart-container {{ margin-bottom: 30px; }}
            .footer {{ margin-top: 40px; color: #777; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>Отчет о бенчмаркинге RAG-системы с различными моделями</h1>
        <p>Дата создания: {timestamp}</p>
        
        <h2>Сводная таблица результатов</h2>
        <table>
            <tr>
                <th>Модель</th>
                <th>Среднее время ответа (сек)</th>
                <th>Успешных запросов (%)</th>
                <th>Покрытие ключевых слов (%)</th>
                <th>Частота цитирования (%)</th>
                <th>Средн. длина ответа (слов)</th>
            </tr>
    """
    
    # Добавляем строки таблицы
    for _, row in summary_table.iterrows():
        html_report += f"""
            <tr>
                <td>{row['Модель']}</td>
                <td>{row['Среднее время ответа (сек)']:.2f}</td>
                <td>{row['Успешных запросов (%)']:.1f}%</td>
                <td>{row['Покрытие ключевых слов (%)']:.1f}%</td>
                <td>{row['Частота цитирования (%)']:.1f}%</td>
                <td>{row['Средн. длина ответа (слов)']:.0f}</td>
            </tr>
        """
    
    # Добавляем графики
    html_report += """
        </table>
        
        <h2>Графики результатов</h2>
        
        <div class="chart-container">
            <h3>Производительность моделей</h3>
            <img src="../benchmark_results/plots/avg_query_time.png" alt="Среднее время выполнения запроса" width="600">
        </div>
        
        <div class="chart-container">
            <h3>Качество ответов</h3>
            <img src="../benchmark_results/rag_quality/plots/keyword_coverage.png" alt="Покрытие ключевых слов" width="600">
        </div>
        
        <div class="chart-container">
            <h3>Частота цитирования</h3>
            <img src="../benchmark_results/rag_quality/plots/citation_rate.png" alt="Частота цитирования" width="600">
        </div>
        
        <div class="chart-container">
            <h3>Комбинированный рейтинг</h3>
            <img src="../benchmark_results/rag_quality/plots/combined_score.png" alt="Комбинированный рейтинг" width="600">
        </div>
        
        <div class="footer">
            <p>Отчет создан автоматически системой бенчмаркинга RAG.</p>
        </div>
    </body>
    </html>
    """
    
    # Сохраняем HTML отчет
    with open(f"{report_dir}/report.html", "w", encoding="utf-8") as f:
        f.write(html_report)
    
    # Сохраняем все данные в JSON
    with open(f"{report_dir}/full_data.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "performance_results": performance_results,
            "quality_results": quality_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nСводный отчет сохранен в {report_dir}/report.html")
    return report_dir

def run_all_benchmarks(models: List[str], mode: str = "qa"):
    """
    Запускает все бенчмарки и создает сводный отчет.
    
    Args:
        models: Список моделей для тестирования
        mode: Режим работы ('qa' или 'compare')
    """
    print("\n========== ЗАПУСК ПОЛНОГО БЕНЧМАРКИНГА ==========")
    print(f"Модели для тестирования: {', '.join(models)}")
    print(f"Режим: {mode}")
    start_time = time.time()
    
    # Создаем директорию для результатов
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_dir = f"benchmark_results/{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # 1. Запуск бенчмарка производительности
    print("\n\n========== БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ ==========")
    performance_results = benchmark_performance(
        model_ids=models,
        mode=mode,
        save_results=True,
        plot_results=True
    )
    
    # 2. Запуск бенчмарка качества
    print("\n\n========== БЕНЧМАРК КАЧЕСТВА ==========")
    quality_results = benchmark_quality(
        model_ids=models,
        mode=mode,
        save_results=True,
        plot_results=True
    )
    
    # 3. Создание сводного отчета
    print("\n\n========== СОЗДАНИЕ СВОДНОГО ОТЧЕТА ==========")
    report_path = create_report(performance_results, quality_results, report_dir)
    
    # Выводим итоговое время
    total_time = time.time() - start_time
    print(f"\nВремя выполнения всех бенчмарков: {total_time:.2f} сек ({total_time/60:.2f} мин)")
    print(f"Отчет сохранен в: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Запуск всех бенчмарков для RAG-системы")
    parser.add_argument("--models", type=str, nargs="+", default=["phi", "qwen"],
                        help="Список моделей для тестирования")
    parser.add_argument("--mode", type=str, choices=["qa", "compare"], default="qa",
                        help="Режим работы (qa или compare)")
    args = parser.parse_args()
    
    run_all_benchmarks(models=args.models, mode=args.mode)

if __name__ == "__main__":
    main() 