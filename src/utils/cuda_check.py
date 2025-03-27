#!/usr/bin/env python
"""
Вспомогательный модуль для проверки доступности CUDA
"""

def check_cuda():
    """
    Проверяет доступность CUDA и выводит информацию о GPU
    
    Returns:
        bool: True если CUDA доступна, False в противном случае
    """
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n=== ИНФОРМАЦИЯ О GPU ===")
            print(f"Доступна CUDA: {torch.cuda.is_available()}")
            print(f"Устройство: {torch.cuda.get_device_name(0)}")
            print(f"Количество GPU: {torch.cuda.device_count()}")
            print(f"Версия CUDA: {torch.version.cuda}")
            
            # Получить дополнительную информацию о памяти GPU
            try:
                print(f"\n=== ПАМЯТЬ GPU ===")
                print(f"Общая память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} ГБ")
                print(f"Выделено: {torch.cuda.memory_allocated(0) / 1024**3:.2f} ГБ")
                print(f"Зарезервировано: {torch.cuda.memory_reserved(0) / 1024**3:.2f} ГБ")
            except Exception as e:
                print(f"Не удалось получить информацию о памяти: {e}")
            
            return True
        else:
            print("CUDA недоступна")
            return False
    except ImportError:
        print("PyTorch не установлен или не поддерживает CUDA")
        return False
    except Exception as e:
        print(f"Ошибка при проверке CUDA: {e}")
        return False


if __name__ == "__main__":
    # Проверка при прямом запуске скрипта
    check_cuda() 