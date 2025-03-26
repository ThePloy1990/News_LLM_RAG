"""
Пакет для бенчмаркинга и оценки производительности RAG-системы.
"""
from src.benchmark.model_performance import run_model_benchmark, run_comparison_benchmark
from src.benchmark.rag_quality import run_rag_evaluation, compare_models_quality
from src.benchmark.run_benchmarks import run_all_benchmarks

__all__ = [
    'run_model_benchmark',
    'run_comparison_benchmark',
    'run_rag_evaluation',
    'compare_models_quality',
    'run_all_benchmarks'
] 