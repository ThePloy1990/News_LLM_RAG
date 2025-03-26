import json
import os
from datetime import datetime
from typing import Dict, List
from rag.rag_pipeline import RAGPipeline


def load_test_queries() -> List[Dict]:
    """Загрузка тестовых запросов."""
    with open("tests/test_queries.json", "r", encoding="utf-8") as f:
        return json.load(f)["queries"]


def save_test_results(results: List[Dict]):
    """Сохранение результатов тестов."""
    os.makedirs("tests/results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tests/results/test_results_{timestamp}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Результаты сохранены в {filename}")


def run_tests():
    """Запуск тестовых запросов."""
    pipeline = RAGPipeline()
    queries = load_test_queries()
    results = []
    
    for query in queries:
        print(f"\nТестирование запроса: {query['text']}")
        
        result = {
            "query_id": query["id"],
            "query_text": query["text"],
            "query_type": query["type"],
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        # Получаем релевантные документы
        docs = pipeline.get_relevant_documents(query["text"])
        result["results"]["relevant_docs"] = [
            {
                "author": doc["author"],
                "source": doc["source"],
                "date": doc["date"]
            }
            for doc in docs
        ]
        
        # Генерируем ответ в зависимости от типа запроса
        if query["type"] == "qa":
            answer = pipeline.generate_qa_response(query["text"])
            result["results"]["qa_response"] = answer
        else:
            analysis = pipeline.compare_opinions(query["text"])
            result["results"]["comparative_analysis"] = analysis
        
        results.append(result)
        print(f"Запрос {query['id']} обработан")
    
    save_test_results(results)
    print("\nТестирование завершено")


if __name__ == "__main__":
    run_tests() 