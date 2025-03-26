"""
Фабрика моделей LLM для RAG-системы
"""
from typing import Optional, Dict, Any
from langchain.llms.base import LLM

from .phi_llm import PhiLLM
from .qwen_llm import QwenLLM

# Удалены импорты моделей с проблемами доступа
# from .saiga_llm import SaigaLLM
# from .gemma_llm import GemmaLLM

class LLMFactory:
    """Фабрика LLM моделей для RAG-системы"""
    
    # Справочник доступных моделей
    AVAILABLE_MODELS = {
        "phi": {
            "name": "Phi-3 Mini (Microsoft)",
            "description": "Легкая и быстрая модель от Microsoft на 3.8B параметров",
            "class": PhiLLM,
            "default_params": {
                "model_name": "microsoft/Phi-3-mini-4k-instruct",  # Исправлено имя модели
                "temperature": 0.7,
                "max_new_tokens": 1024,
                "repetition_penalty": 1.1
            }
        },
        "qwen": {
            "name": "Qwen 7B (Alibaba)",
            "description": "Мощная многоязычная модель от Alibaba",
            "class": QwenLLM,
            "default_params": {
                "model_name": "Qwen/Qwen1.5-7B-Chat",
                "temperature": 0.7,
                "max_new_tokens": 1024,
                "repetition_penalty": 1.1
            }
        }
        # Saiga удалена, т.к. требует авторизации и недоступна
    }
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Возвращает список доступных моделей с описанием"""
        return {
            model_id: {
                "name": model_info["name"],
                "description": model_info["description"]
            } for model_id, model_info in cls.AVAILABLE_MODELS.items()
        }
    
    @classmethod
    def get_model(cls, model_id: str, **kwargs) -> Optional[LLM]:
        """
        Создает и возвращает экземпляр модели по указанному ID
        
        Args:
            model_id: ID модели ("phi", "qwen")
            **kwargs: дополнительные параметры для инициализации модели
                
        Returns:
            Инициализированная LLM модель или None, если ID не найден
        """
        if model_id not in cls.AVAILABLE_MODELS:
            print(f"Модель {model_id} не найдена. Доступные модели: {', '.join(cls.AVAILABLE_MODELS.keys())}")
            return None
        
        model_info = cls.AVAILABLE_MODELS[model_id]
        model_class = model_info["class"]
        model_params = model_info["default_params"].copy()
        
        # Обновляем параметры по умолчанию теми, что переданы в kwargs
        model_params.update(kwargs)
        
        try:
            print(f"Инициализация модели {model_info['name']}...")
            model = model_class(**model_params)
            return model
        except Exception as e:
            print(f"Ошибка при инициализации модели {model_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Если не удалось инициализировать запрошенную модель, используем Phi как запасной вариант
            if model_id != "phi" and "phi" in cls.AVAILABLE_MODELS:
                print(f"Не удалось инициализировать модель '{model_id}', использую Phi как запасной вариант")
                return cls.get_model("phi")
            
            return None 