import click
from rag.rag_pipeline import RAGPipeline
from data.fetch_twitter import main as fetch_twitter
from data.prepare_data import main as prepare_data
from rag.build_index import main as build_index
from benchmark.run_benchmarks import run_all_benchmarks


@click.group()
def cli():
    """CLI для системы анализа новостей."""
    pass


@cli.command()
def collect_data():
    """Сбор данных из Twitter."""
    click.echo("Сбор данных из Twitter...")
    fetch_twitter()
    
    click.echo("Подготовка данных...")
    prepare_data()
    
    click.echo("Сбор данных завершен")


@cli.command()
def update_index():
    """Обновление векторного индекса."""
    click.echo("Построение индекса...")
    build_index()
    click.echo("Индекс обновлен")


@cli.command()
@click.option("--mode", type=click.Choice(["qa", "compare"]), default="qa",
              help="Режим работы: qa - ответы на вопросы, compare - сравнительный анализ")
@click.option("--query", prompt="Введите ваш запрос",
              help="Текст запроса для анализа")
def analyze(mode, query):
    """Анализ данных с использованием RAG."""
    pipeline = RAGPipeline()
    
    if mode == "qa":
        click.echo("\n=== Ответ на вопрос ===")
        answer = pipeline.generate_qa_response(query)
        click.echo(answer)
    else:
        click.echo("\n=== Сравнительный анализ ===")
        analysis = pipeline.compare_opinions(query)
        click.echo(analysis)


@cli.command()
@click.option("--query", prompt="Введите ваш запрос",
              help="Текст запроса для поиска релевантных документов")
def search(query):
    """Поиск релевантных документов."""
    pipeline = RAGPipeline()
    docs = pipeline.get_relevant_documents(query)
    
    click.echo("\n=== Релевантные документы ===")
    for doc in docs:
        click.echo(f"\nАвтор: {doc['author']}")
        click.echo(f"Дата: {doc['date']}")
        if 'url' in doc:
            click.echo(f"URL: {doc['url']}")
        click.echo("---")


@cli.command()
@click.option("--models", "-m", multiple=True, default=["phi", "qwen"],
              help="Модели для тестирования (можно указать несколько)")
@click.option("--mode", type=click.Choice(["qa", "compare"]), default="qa",
              help="Режим работы: qa - ответы на вопросы, compare - сравнительный анализ")
def benchmark(models, mode):
    """Запуск бенчмаркинга для сравнения моделей."""
    click.echo(f"Запуск бенчмаркинга для моделей: {', '.join(models)} в режиме {mode}")
    
    models_list = list(models)  # преобразуем кортеж в список для передачи функции
    
    # Запускаем полный бенчмарк
    run_all_benchmarks(models=models_list, mode=mode)
    
    click.echo("Бенчмаркинг завершен. Результаты сохранены в директории benchmark_results")


if __name__ == "__main__":
    cli() 