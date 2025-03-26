#!/usr/bin/env python
"""
Модуль для оценки качества RAG-системы.
Измеряет релевантность, точность и другие метрики качества ответов.
"""
import os
import json
import argparse
import time
import csv
import numpy as np
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.rag.llm_factory import LLMFactory
from src.rag.rag_pipeline import RAGPipeline

# Загружаем переменные окружения
load_dotenv()

# Тестовый набор вопросов с эталонными ответами
EVALUATION_QUESTIONS = [
    {
        "id": "q1",
        "question": "Какие перспективы развития Ethereum в 2023-2024 годах по мнению Виталика Бутерина?",
        "ideal_answer_keywords": [
            "масштабирование", "sharding", "rollups", "L2", "обновление", "ETH 2.0", 
            "Proof of Stake", "снижение комиссий", "EIP", "Danksharding"
        ],
        "reference_sources": ["VitalikButerin", "gavofyork"]
    },
    {
        "id": "q2", 
        "question": "Какие факторы влияют на цену Bitcoin по мнению Виталика Бутерина?",
        "ideal_answer_keywords": [
            "предложение", "спрос", "институциональные инвесторы", "регулирование",
            "халвинг", "майнинг", "принятие", "ETF", "волатильность", "макроэкономика"
        ],
        "reference_sources": ["100trillionUSD", "novogratz", "woonomic"]
    },
    {
        "id": "q3",
        "question": "Какая самая вкусная пицца в Москве?",
        "ideal_answer_keywords": [
            "нерелевантный запрос", "вне тематики", "криптовалюты", "блокчейн",
            "не относится", "невозможно ответить", "другая тема", "нет данных"
        ],
        "reference_sources": ["PeterLBrandt", "RaoulGMI", "APompliano"]
    },
    {
        "id": "q4",
        "question": "Какие упомянания смарт-контрактов в твиттере за март 2025?",
        "ideal_answer_keywords": [
            "самоисполняемый", "программа", "блокчейн", "условия", "автоматизация", 
            "DeFi", "NFT", "токенизация", "децентрализованные приложения", "Solidity"
        ],
        "reference_sources": ["VitalikButerin", "cdixon"]
    },
    {
        "id": "q5",
        "question": "Какие упомянания DeFi в твиттере за март 2025?",
        "ideal_answer_keywords": [
            "децентрализация", "без посредников", "прозрачность", "доступность", 
            "программируемость", "открытый код", "комбинируемость", "composability", 
            "автоматизация", "yield farming"
        ],
        "reference_sources": ["0xfoobar", "DefiPulse"]
    },
    {
        "id": "q6",
        "question": "Какие темы поднимал Виталик Бутерин в своих твитах за март 2025?",
        "ideal_answer_keywords": [
            "L2", "Layer 2", "Rollups", "ZK-Rollups", "Optimistic Rollups", "Sidechains", 
            "шардинг", "Plasma", "State Channels", "Ethereum 2.0"
        ],
        "reference_sources": ["VitalikButerin", "gavofyork"]
    }
]

def load_embedding_model():
    """Загружает модель для эмбеддингов."""
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("Модель эмбеддингов успешно загружена")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели эмбеддингов: {e}")
        return None

def calculate_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Вычисляет косинусную схожесть между двумя текстами."""
    try:
        emb1 = model.encode([text1])[0]
        emb2 = model.encode([text2])[0]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    except Exception as e:
        print(f"Ошибка при вычислении схожести: {e}")
        return 0.0

def calculate_keyword_coverage(answer: str, keywords: List[str]) -> Tuple[float, List[str]]:
    """Вычисляет покрытие ключевых слов в ответе."""
    answer_lower = answer.lower()
    found_keywords = []
    
    for keyword in keywords:
        if keyword.lower() in answer_lower:
            found_keywords.append(keyword)
    
    coverage = len(found_keywords) / len(keywords) if keywords else 0
    return coverage, found_keywords

def has_source_citation(answer: str, sources: List[str]) -> Tuple[bool, List[str]]:
    """Проверяет, содержит ли ответ упоминания указанных источников."""
    answer_lower = answer.lower()
    cited_sources = []
    
    for source in sources:
        if source.lower() in answer_lower:
            cited_sources.append(source)
    
    return len(cited_sources) > 0, cited_sources

def evaluate_answer(question: Dict, answer: str, embedding_model: SentenceTransformer) -> Dict:
    """Оценивает ответ по нескольким критериям."""
    # Вычисляем покрытие ключевых слов
    keyword_coverage, found_keywords = calculate_keyword_coverage(
        answer, question["ideal_answer_keywords"]
    )
    
    # Проверяем упоминание источников
    has_citations, cited_sources = has_source_citation(
        answer, question.get("reference_sources", [])
    )
    
    # Оцениваем общее качество ответа (на основе покрытия ключевых слов)
    quality_score = keyword_coverage
    
    # Длина ответа (символы и слова)
    char_count = len(answer)
    word_count = len(answer.split())
    
    # Результаты оценки
    evaluation = {
        "question_id": question["id"],
        "keyword_coverage": keyword_coverage,
        "keyword_coverage_percent": keyword_coverage * 100,
        "found_keywords": found_keywords,
        "missing_keywords": [k for k in question["ideal_answer_keywords"] if k not in found_keywords],
        "has_citations": has_citations,
        "cited_sources": cited_sources,
        "quality_score": quality_score,
        "char_count": char_count,
        "word_count": word_count
    }
    
    return evaluation

def run_rag_evaluation(model_id: str, mode: str = "qa", questions: List[Dict] = None, 
                     verbose: bool = True, save_results: bool = True) -> Dict[str, Any]:
    """
    Запускает оценку качества RAG системы.
    
    Args:
        model_id: Идентификатор модели для тестирования
        mode: Режим работы ('qa' или 'compare')
        questions: Список вопросов для тестирования
        verbose: Подробный вывод результатов
        save_results: Сохранять результаты в файл
    
    Returns:
        Словарь с результатами оценки
    """
    if questions is None:
        questions = EVALUATION_QUESTIONS
    
    # Инициализируем модель эмбеддингов
    embedding_model = load_embedding_model()
    
    if embedding_model is None:
        return {"error": "Не удалось загрузить модель эмбеддингов"}
    
    # Создаем пайплайн с указанной моделью
    if verbose:
        print(f"\n=== Оценка качества RAG с моделью: {model_id} ===")
    
    pipeline = RAGPipeline(model_id=model_id)
    
    # Результаты по вопросам
    results = []
    all_answers = []
    
    # Обработка каждого вопроса
    for question in tqdm(questions, desc="Оценка вопросов"):
        if verbose:
            print(f"\nВопрос [{question['id']}]: {question['question']}")
        
        # Генерируем ответ
        start_time = time.time()
        
        try:
            if mode == "qa":
                answer = pipeline.generate_qa_response(question["question"])
            else:
                answer = pipeline.compare_opinions(question["question"])
            
            # Время генерации
            generation_time = time.time() - start_time
            
            # Оцениваем ответ
            evaluation = evaluate_answer(question, answer, embedding_model)
            
            # Добавляем время и статус
            evaluation.update({
                "question": question["question"],
                "answer": answer,
                "generation_time": generation_time,
                "success": True
            })
            
            if verbose:
                print(f"Покрытие ключевых слов: {evaluation['keyword_coverage_percent']:.1f}%")
                print(f"Найденные ключевые слова: {', '.join(evaluation['found_keywords'])}")
                print(f"Цитирует источники: {'Да' if evaluation['has_citations'] else 'Нет'}")
                if evaluation["has_citations"]:
                    print(f"Упомянутые источники: {', '.join(evaluation['cited_sources'])}")
                print(f"Длина ответа: {evaluation['char_count']} символов, {evaluation['word_count']} слов")
                print(f"Время генерации: {generation_time:.2f} сек")
            
            results.append(evaluation)
            all_answers.append(answer)
            
        except Exception as e:
            if verbose:
                print(f"Ошибка при обработке вопроса: {e}")
            
            results.append({
                "question_id": question["id"],
                "question": question["question"],
                "success": False,
                "error": str(e)
            })
    
    # Вычисляем средние показатели
    successful_results = [r for r in results if r.get("success", False)]
    
    if successful_results:
        avg_keyword_coverage = np.mean([r["keyword_coverage"] for r in successful_results])
        avg_generation_time = np.mean([r["generation_time"] for r in successful_results])
        citation_rate = np.mean([1 if r["has_citations"] else 0 for r in successful_results])
        avg_char_count = np.mean([r["char_count"] for r in successful_results])
        avg_word_count = np.mean([r["word_count"] for r in successful_results])
    else:
        avg_keyword_coverage = 0
        avg_generation_time = 0
        citation_rate = 0
        avg_char_count = 0
        avg_word_count = 0
    
    # Финальные результаты
    summary = {
        "model_id": model_id,
        "mode": mode,
        "total_questions": len(questions),
        "successful_questions": len(successful_results),
        "success_rate": len(successful_results) / len(questions) if questions else 0,
        "avg_keyword_coverage": avg_keyword_coverage,
        "avg_keyword_coverage_percent": avg_keyword_coverage * 100,
        "citation_rate": citation_rate,
        "citation_rate_percent": citation_rate * 100,
        "avg_generation_time": avg_generation_time,
        "avg_char_count": avg_char_count,
        "avg_word_count": avg_word_count,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if verbose:
        print("\n=== Итоговые результаты ===")
        print(f"Успешно обработано: {summary['successful_questions']}/{summary['total_questions']} вопросов")
        print(f"Среднее покрытие ключевых слов: {summary['avg_keyword_coverage_percent']:.1f}%")
        print(f"Частота цитирования: {summary['citation_rate_percent']:.1f}%")
        print(f"Среднее время генерации: {summary['avg_generation_time']:.2f} сек")
        print(f"Средняя длина ответа: {summary['avg_char_count']:.0f} символов, {summary['avg_word_count']:.0f} слов")
    
    # Сохраняем результаты
    if save_results:
        os.makedirs("benchmark_results/rag_quality", exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем итоговый отчет
        with open(f"benchmark_results/rag_quality/summary_{model_id}_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "detailed_results": results
            }, f, ensure_ascii=False, indent=2)
        
        # Сохраняем вопросы и ответы в CSV
        with open(f"benchmark_results/rag_quality/qa_pairs_{model_id}_{timestamp}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Вопрос", "Ответ", "Покрытие ключевых слов (%)", "Время генерации (сек)"])
            
            for result in results:
                if result.get("success", False):
                    writer.writerow([
                        result["question_id"],
                        result["question"],
                        result["answer"],
                        f"{result.get('keyword_coverage_percent', 0):.1f}",
                        f"{result.get('generation_time', 0):.2f}"
                    ])
        
        if verbose:
            print(f"Результаты сохранены в директории benchmark_results/rag_quality/")
    
    return {
        "summary": summary,
        "results": results
    }

def compare_models_quality(model_ids: List[str], mode: str = "qa", 
                         questions: List[Dict] = None,
                         save_results: bool = True,
                         plot_results: bool = True):
    """
    Сравнивает качество ответов разных моделей в RAG-системе.
    
    Args:
        model_ids: Список идентификаторов моделей для тестирования
        mode: Режим работы ('qa' или 'compare')
        questions: Список вопросов для тестирования
        save_results: Сохранять результаты в файл
        plot_results: Строить графики результатов
    """
    if questions is None:
        questions = EVALUATION_QUESTIONS
    
    print("\n===== Сравнение качества RAG-системы с разными моделями =====")
    print(f"Модели для тестирования: {', '.join(model_ids)}")
    print(f"Количество вопросов: {len(questions)}")
    print(f"Режим: {mode}")
    
    all_results = []
    
    for model_id in model_ids:
        evaluation = run_rag_evaluation(model_id, mode, questions, verbose=True, save_results=True)
        all_results.append(evaluation["summary"])
    
    # Сохраняем сводные результаты
    if save_results:
        os.makedirs("benchmark_results/rag_quality", exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results/rag_quality/models_comparison_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"Сравнительные результаты сохранены в {filename}")
    
    # Строим графики
    if plot_results:
        plot_quality_comparison(all_results)

def plot_quality_comparison(results: List[Dict]):
    """Строит графики сравнения качества RAG-системы с разными моделями."""
    if not results:
        return
    
    # Создаем DataFrame для удобного анализа
    df = pd.DataFrame(results)
    
    # Создаем директорию для графиков
    os.makedirs("benchmark_results/rag_quality/plots", exist_ok=True)
    
    # График 1: Покрытие ключевых слов
    plt.figure(figsize=(10, 6))
    coverage_by_model = df.sort_values(by="avg_keyword_coverage_percent")["avg_keyword_coverage_percent"]
    coverage_by_model.plot(kind="bar", color="lightblue")
    plt.title("Среднее покрытие ключевых слов (%)")
    plt.ylabel("Покрытие (%)")
    plt.xlabel("Модель")
    plt.xticks(range(len(df)), df["model_id"], rotation=45)
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig("benchmark_results/rag_quality/plots/keyword_coverage.png")
    
    # График 2: Частота цитирования
    plt.figure(figsize=(10, 6))
    citation_by_model = df.sort_values(by="citation_rate_percent")["citation_rate_percent"]
    citation_by_model.plot(kind="bar", color="salmon")
    plt.title("Частота цитирования источников (%)")
    plt.ylabel("Частота (%)")
    plt.xlabel("Модель")
    plt.xticks(range(len(df)), df["model_id"], rotation=45)
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig("benchmark_results/rag_quality/plots/citation_rate.png")
    
    # График 3: Время генерации
    plt.figure(figsize=(10, 6))
    time_by_model = df.sort_values(by="avg_generation_time")["avg_generation_time"]
    time_by_model.plot(kind="bar", color="lightgreen")
    plt.title("Среднее время генерации (сек)")
    plt.ylabel("Время (сек)")
    plt.xlabel("Модель")
    plt.xticks(range(len(df)), df["model_id"], rotation=45)
    plt.tight_layout()
    plt.savefig("benchmark_results/rag_quality/plots/generation_time.png")
    
    # График 4: Комбинированный рейтинг (можно настроить веса)
    plt.figure(figsize=(10, 6))
    
    # Нормализуем метрики для сравнения
    max_time = df["avg_generation_time"].max()
    df["time_score"] = 1 - (df["avg_generation_time"] / max_time)  # Инвертируем, чтобы меньшее время было лучше
    
    # Вычисляем комбинированный рейтинг (можно настроить веса)
    coverage_weight = 0.5
    citation_weight = 0.3
    time_weight = 0.2
    
    df["combined_score"] = (
        (df["avg_keyword_coverage"] * coverage_weight) +
        (df["citation_rate"] * citation_weight) +
        (df["time_score"] * time_weight)
    )
    
    # Сортируем и рисуем
    combined_score = df.sort_values(by="combined_score", ascending=False)["combined_score"]
    combined_score.plot(kind="bar", color="mediumpurple")
    plt.title("Комбинированный рейтинг моделей")
    plt.ylabel("Рейтинг (0-1)")
    plt.xlabel("Модель")
    plt.xticks(range(len(df)), df.loc[combined_score.index, "model_id"], rotation=45)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig("benchmark_results/rag_quality/plots/combined_score.png")
    
    print(f"Графики сохранены в директории benchmark_results/rag_quality/plots/")
    
    # Возвращаем лучшую модель по комбинированному рейтингу
    best_model = df.loc[combined_score.idxmax(), "model_id"]
    print(f"\nЛучшая модель по комбинированному рейтингу: {best_model}")
    return best_model

def main():
    parser = argparse.ArgumentParser(description="Оценка качества RAG-системы")
    parser.add_argument("--models", type=str, nargs="+", default=["phi", "qwen"],
                        help="Список моделей для тестирования")
    parser.add_argument("--mode", type=str, choices=["qa", "compare"], default="qa",
                        help="Режим работы (qa или compare)")
    args = parser.parse_args()
    
    print("\n===== Оценка качества RAG-системы =====")
    print(f"Режим: {args.mode}")
    print(f"Модели для тестирования: {', '.join(args.models)}")
    
    compare_models_quality(
        model_ids=args.models,
        mode=args.mode,
        save_results=True,
        plot_results=True
    )

if __name__ == "__main__":
    main() 