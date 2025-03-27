"""
Модуль для работы с моделью Qwen от Alibaba
"""
import os
import torch
from typing import Any, List, Mapping, Optional, Dict
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from pydantic import Field
import re
import json

class QwenLLM(LLM):
    """LLM обертка для модели Qwen от Alibaba"""
    
    model_name: str = Field("Qwen/Qwen1.5-7B-Chat")
    temperature: float = Field(0.7)
    max_new_tokens: int = Field(1024)
    repetition_penalty: float = Field(1.1)
    system_prompt: str = Field("""Ты - опытный аналитик криптовалютного рынка с глубоким пониманием технологий блокчейн. 
Твоя задача - анализировать информацию из различных источников и предоставлять структурированные ответы по вопросам криптовалют.
Всегда опирайся только на факты из предоставленных источников.""")
    
    tokenizer: Any = None
    model: Any = None
    device: str = None
    
    def __init__(self, **kwargs):
        """Инициализация модели Qwen"""
        super().__init__(**kwargs)
        
        # Определяем устройство
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Устройство для Qwen: {self.device}")
        
        if self.device == "cuda":
            # Вывод информации о GPU
            print(f"Доступно GPU: {torch.cuda.device_count()}")
            print(f"Активная GPU: {torch.cuda.get_device_name(0)}")
            print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} ГБ")
        
        try:
            print(f"Загрузка модели {self.model_name}...")
            
            # Загружаем модель и токенизатор напрямую
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Отптимизированная загрузка модели для RTX 4090
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",  # Автоматическое распределение слоев по доступным GPU
                    torch_dtype=torch.float16,  # Используем float16 для большей производительности
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,  # Снижаем использование CPU памяти
                    # 4-битное квантование для Qwen (если доступно)
                    quantization_config={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16} 
                    if hasattr(torch, "bfloat16") else None
                )
                # Оптимизации CUDA
                torch.backends.cudnn.benchmark = True
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
            
            print(f"Модель {self.model_name} успешно загружена")
            
        except Exception as e:
            print(f"Ошибка при инициализации модели: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    @property
    def _llm_type(self) -> str:
        return "qwen"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Генерация текста с помощью модели Qwen."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Модель не инициализирована")
        
        try:
            # Подготавливаем входные данные в формате чата Qwen
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Формируем шаблон чата
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Замеряем время генерации
            start_time = time.time()
            
            # Токенизируем вход
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            # Очистка памяти CUDA перед генерацией
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Генерируем ответ
            with torch.no_grad():
                # Параметры генерации
                generation_config = {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "repetition_penalty": self.repetition_penalty,
                    "do_sample": True,
                    "use_cache": True,  # Включаем кеширование для скорости
                    "pad_token_id": self.tokenizer.eos_token_id
                }
                
                # Не передаём attention_mask дважды
                # Из inputs уже берётся attention_mask
                outputs = self.model.generate(**inputs, **generation_config)
            
            try:
                # Декодируем и получаем только новые токены
                input_length = inputs["input_ids"].shape[1]
                response = self.tokenizer.decode(
                    outputs[0][input_length:], 
                    skip_special_tokens=True
                ).strip()
            except Exception as extract_error:
                print(f"Ошибка при извлечении ответа: {extract_error}")
                # Запасной вариант - декодируем весь выход и убираем вход
                full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = full_output.replace(input_text, "").strip()
            
            end_time = time.time()
            print(f"Время генерации Qwen: {end_time - start_time:.2f} сек")
            
            # Применяем stop-последовательности, если они заданы
            if stop:
                for sequence in stop:
                    if sequence in response:
                        response = response.split(sequence)[0]
            
            return response
            
        except Exception as e:
            print(f"Ошибка при генерации текста: {e}")
            import traceback
            traceback.print_exc()
            return f"Произошла ошибка при генерации ответа: {str(e)[:200]}"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Возвращает параметры, идентифицирующие LLM."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
        } 