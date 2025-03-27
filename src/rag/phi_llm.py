"""
Модуль для работы с моделью Phi-3 от Microsoft
"""
import os
import torch
from typing import Any, List, Mapping, Optional, Dict
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import time
from pydantic import Field
import re
import json

class PhiLLM(LLM):
    """LLM обертка для модели Phi от Microsoft"""
    
    model_name: str = Field("microsoft/Phi-3-mini-4k-instruct")
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
        """Инициализация модели Phi"""
        super().__init__(**kwargs)
        
        # Определяем устройство
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Устройство для Phi: {self.device}")
        
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
            
            # При загрузке модели указываем дополнительные параметры для оптимизации CUDA
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,  # Используем float16 для экономии памяти на RTX 4090
                    trust_remote_code=True,
                    low_cpu_mem_usage=True     # Снижаем использование CPU памяти
                )
                # Устанавливаем CUDA оптимизации
                torch.backends.cudnn.benchmark = True  # Оптимизация для постоянных размеров входов
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
        return "phi"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Генерация текста с помощью модели Phi."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Модель не инициализирована")
        
        try:
            # Phi-3 использует формат <|user|>, <|assistant|>
            input_text = f"<|user|>\n{prompt}\n<|assistant|>\n"
            
            # Замеряем время генерации
            start_time = time.time()
            
            # Токенизируем вход
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            # Генерируем ответ напрямую через модель
            with torch.no_grad():
                if self.device == "cuda":
                    # Очищаем кэш CUDA перед генерацией для освобождения памяти
                    torch.cuda.empty_cache()
                
                # Создаем конфигурацию генерации без использования DynamicCache.get_max_length()
                generation_config = GenerationConfig(
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
                )
                
                try:
                    # Пробуем безопасный способ генерации без кэша
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        generation_config=generation_config
                    )
                except AttributeError as e:
                    if 'get_max_length' in str(e):
                        # Запасной вариант без использования past_key_values
                        print("Используем упрощенную генерацию без кэша для Phi-3")
                        outputs = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask", None),
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            repetition_penalty=self.repetition_penalty,
                            do_sample=True,
                            use_cache=False  # Отключаем кэш полностью
                        )
                    else:
                        raise
            
            # Декодируем выход (полностью)
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Извлекаем ответ ассистента (используем более надежный метод)
            try:
                # Пытаемся извлечь ответ по тегам
                if "<|assistant|>" in full_output:
                    response_parts = full_output.split("<|assistant|>")
                    if len(response_parts) > 1:
                        assistant_response = response_parts[1].strip()
                        if "<|end|>" in assistant_response:
                            assistant_response = assistant_response.split("<|end|>")[0].strip()
                    else:
                        # Если не удалось разделить по ассистенту, берем всё после запроса
                        assistant_response = full_output.replace(input_text, "").strip()
                else:
                    # Запасной вариант - берем токены, которые не были во входе
                    input_length = inputs["input_ids"].shape[1]
                    assistant_response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            except Exception as extract_error:
                print(f"Ошибка при извлечении ответа: {extract_error}")
                # Последний вариант - берем всё после последнего знакомого маркера
                assistant_response = full_output.split("\n<|assistant|>\n")[-1].strip()
            
            end_time = time.time()
            print(f"Время генерации: {end_time - start_time:.2f} сек")
            
            # Применяем stop-последовательности, если они заданы
            if stop:
                for sequence in stop:
                    if sequence in assistant_response:
                        assistant_response = assistant_response.split(sequence)[0]
            
            return assistant_response.strip()
            
        except Exception as e:
            print(f"Ошибка при генерации текста: {e}")
            import traceback
            traceback.print_exc()
            
            # Возвращаем более информативное сообщение об ошибке
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