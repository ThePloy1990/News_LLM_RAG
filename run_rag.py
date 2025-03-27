#!/usr/bin/env python
"""
Скрипт для запуска RAG-системы с выбором модели
"""
import os
import argparse
from dotenv import load_dotenv
from src.rag.llm_factory import LLMFactory
from src.rag.rag_pipeline import RAGPipeline
from src.utils.cuda_check import check_cuda

def list_models():
    """Выводит список доступных моделей"""
    print("\n=== Доступные модели ===")
    models = LLMFactory.list_available_models()
    for model_id, info in models.items():
        print(f"{model_id.ljust(8)} - {info['name']}: {info['description']}")
    print()

def main():
    """Основная функция запуска RAG-системы"""
    # Загружаем переменные окружения
    load_dotenv()
    
    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(description="RAG-система для анализа криптовалютных новостей")
    parser.add_argument("--model", "-m", type=str, default="phi",
                      help=f"Модель (доступны: {', '.join(LLMFactory.AVAILABLE_MODELS.keys())})")
    parser.add_argument("--query", "-q", type=str,
                      help="Запрос для анализа (если не указан, будет предложено ввести)")
    parser.add_argument("--mode", type=str, default="qa", choices=["qa", "compare"],
                      help="Режим анализа: qa (вопрос-ответ) или compare (сравнение мнений)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                      help="Температура генерации (0.0-1.0)")
    parser.add_argument("--list-models", "-l", action="store_true",
                      help="Показать список доступных моделей и выйти")
    parser.add_argument("--index-path", type=str, default="faiss_index",
                      help="Путь к индексу векторной базы данных")
    parser.add_argument("--use-cuda", action="store_true", default=True,
                      help="Использовать CUDA для ускорения (если доступно)")
    parser.add_argument("--no-cuda", action="store_true",
                      help="Не использовать CUDA даже если доступно")
    parser.add_argument("--check-cuda", action="store_true",
                      help="Проверить доступность CUDA и выйти")
    
    # Парсим аргументы
    args = parser.parse_args()
    
    # Проверяем доступность CUDA, если запрошено
    if args.check_cuda:
        check_cuda()
        return
    
    # Показываем список моделей и выходим, если требуется
    if args.list_models:
        list_models()
        return
    
    # Проверяем, что ID модели валидный
    if args.model not in LLMFactory.AVAILABLE_MODELS:
        print(f"Ошибка: Модель '{args.model}' не найдена.")
        list_models()
        return
    
    # Определяем, использовать ли CUDA
    use_cuda = args.use_cuda and not args.no_cuda
    if use_cuda:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"CUDA доступна: {torch.cuda.get_device_name(0)}")
                # Настраиваем переменные окружения для оптимальной работы CUDA
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            else:
                print("CUDA недоступна, будет использован CPU")
                use_cuda = False
        except ImportError:
            print("Не удалось импортировать torch, будет использован CPU")
            use_cuda = False
    else:
        print("CUDA отключена пользователем, будет использован CPU")
    
    # Запрашиваем ввод запроса, если он не был предоставлен
    query = args.query
    if not query:
        print(f"Введите запрос для {args.mode} анализа:")
        query = input("> ")
    
    # Отображаем выбранные параметры
    model_info = LLMFactory.AVAILABLE_MODELS[args.model]
    print("\n=== Параметры запуска ===")
    print(f"Модель:      {model_info['name']}")
    print(f"Запрос:      {query}")
    print(f"Режим:       {'Вопрос-ответ' if args.mode == 'qa' else 'Сравнительный анализ'}")
    print(f"Температура: {args.temperature}")
    print(f"Индекс:      {args.index_path}")
    print(f"CUDA:        {'Включена' if use_cuda else 'Отключена'}")
    print("=" * 40)
    
    # Принудительно отключаем GPU для FAISS из-за проблем с совместимостью
    try:
        # Инициализируем пайплайн с выбранной моделью, но отключаем GPU для FAISS
        pipeline = RAGPipeline(
            index_path=args.index_path,
            model_id=args.model,
            temperature=args.temperature,
            use_gpu=False  # Принудительно отключаем GPU для FAISS
        )
        
        # Выполняем запрос в зависимости от режима
        if args.mode == "qa":
            print("\n=== Ответ на вопрос ===")
            answer = pipeline.generate_qa_response(query)
        else:
            print("\n=== Сравнительный анализ ===")
            answer = pipeline.compare_opinions(query)
        
        print(answer)
        
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 