import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
# Импортируем FAISS с обработкой возможной ошибки
try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    print("ПРЕДУПРЕЖДЕНИЕ: Не удалось импортировать FAISS. Будет использована упрощенная версия векторного хранилища.")
    FAISS_AVAILABLE = False
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import re
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore

from .llm_factory import LLMFactory
from .models import RAGResponse

# Загрузка переменных окружения
load_dotenv()

# Список ключевых слов для фильтрации по криптовалютной тематике
CRYPTO_KEYWORDS = [
    # Основные криптовалюты
    'биткоин', 'bitcoin', 'btc', 'eth', 'ethereum', 'эфириум',
    'криптовалюта', 'crypto', 'блокчейн', 'blockchain',
    
    # Альткоины и токены
    'альткоин', 'altcoin', 'токен', 'token', 'nft', 'defi', 'dao',
    'solana', 'sol', 'cardano', 'ada', 'binance', 'bnb', 'xrp', 'ripple',
    'polkadot', 'dot', 'avalanche', 'avax', 'tether', 'usdt', 'stablecoin',
    
    # Технологии и концепции
    'майнинг', 'mining', 'стейкинг', 'staking', 'proof of stake', 'pos',
    'proof of work', 'pow', 'consensus', 'консенсус', 'смарт-контракт',
    'smart contract', 'layer 2', 'l2', 'rollup', 'scaling', 'scalability',
    'шардинг', 'sharding', 'web3', 'веб3', 'dapp', 'децентрализация', 
    'decentralization', 'децентрализованный', 'centralized', 'централизованный',
    'p2p', 'пиринговый', 'cross-chain', 'кроссчейн', 'interoperability',
    
    # Обмен и торговля
    'биржа', 'exchange', 'dex', 'swap', 'amm', 'liquidity', 'ликвидность',
    'trading', 'трейдинг', 'market', 'рынок', 'спот', 'spot', 'futures', 'фьючерсы',
    'options', 'опционы', 'волатильность', 'volatility', 'bull', 'медвежий',
    'bear', 'бычий', 'pump', 'dump', 'yield', 'farming', 'майнер', 'miner',
    
    # Регулирование и экономика
    'регулирование', 'regulation', 'sec', 'цб', 'центробанк', 'санкции',
    'sanctions', 'compliance', 'kyc', 'aml', 'налог', 'tax', 'законопроект',
    'запрет', 'ban', 'легализация', 'adoption', 'adoption', 'цифровая экономика',
    'цифровой рубль', 'cbdc', 'инфляция', 'inflation', 'hedge', 'хедж',
    
    # Персоналии и организации
    'vitalik', 'buterin', 'виталик', 'бутерин', 'satoshi', 'nakamoto', 'сатоши',
    'накамото', 'chainanlysis', 'consensys', 'grayscale', 'microstrategy', 'saylor',
    'cz', 'чанпэн', 'binance', 'бинанс', 'coinbase', 'kraken', 'ftx', 'sbf',
    'bankman-fried', 'celsius', 'metamask', 'opensea', 'uniswap', 'aave',
    'compound', 'makerdao', 'arbitrum', 'optimism'
]

class Opinion(BaseModel):
    """Модель для структурированного вывода мнений."""
    author: str = Field(description="Автор мнения")
    sentiment: str = Field(description="Тональность мнения (позитивная/негативная/нейтральная)")
    key_points: List[str] = Field(description="Ключевые аргументы автора")
    confidence: float = Field(description="Уверенность в анализе (от 0 до 1)")


# Реализация простого векторного хранилища, если FAISS недоступен
class SimpleVectorStore:
    """Простая замена FAISS для случаев, когда FAISS недоступен"""
    def __init__(self, embeddings, documents):
        """Инициализация простого векторного хранилища"""
        self.embeddings = embeddings
        self.documents = documents if documents else []
        self.document_embeddings = []
        
        # Подготавливаем эмбеддинги для документов
        if documents:
            texts = [doc.page_content for doc in documents]
            self.document_embeddings = embeddings.embed_documents(texts)
    
    def similarity_search(self, query, k=5):
        """Поиск k наиболее похожих документов"""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Если документов нет, возвращаем пустой список
        if not self.documents:
            return []
        
        # Получаем эмбеддинг для запроса
        query_embedding = self.embeddings.embed_query(query)
        
        # Преобразуем в numpy массивы
        query_embedding_np = np.array(query_embedding).reshape(1, -1)
        document_embeddings_np = np.array(self.document_embeddings)
        
        # Рассчитываем косинусное сходство
        similarities = cosine_similarity(query_embedding_np, document_embeddings_np).flatten()
        
        # Получаем индексы k наиболее похожих документов
        indices = np.argsort(similarities)[::-1][:k]
        
        # Возвращаем наиболее похожие документы
        return [self.documents[i] for i in indices]
    
    @classmethod
    def load_local(cls, folder_path, embeddings, allow_dangerous_deserialization=False):
        """Загрузка из локальной директории"""
        import pickle
        import os
        
        # Загружаем документы из файла
        docs_path = os.path.join(folder_path, "documents.pkl")
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                documents = pickle.load(f)
        else:
            print("Предупреждение: файл documents.pkl не найден")
            documents = []
        
        return cls(embeddings, documents)


class RAGPipeline:
    def __init__(self, index_path: str = "faiss_index", model_id: str = "phi", use_gpu: bool = True, **model_kwargs):
        """
        Инициализация RAG-пайплайна.
        
        Args:
            index_path: Путь к индексу FAISS
            model_id: ID языковой модели ('phi', 'gemma', 'qwen', 'saiga')
            use_gpu: Использовать ли GPU для FAISS (если доступно)
            **model_kwargs: Дополнительные параметры для инициализации языковой модели
        """
        # Используем SentenceTransformers для эмбеддингов
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Проверяем доступность CUDA для FAISS
        self.use_gpu = False  # Принудительно отключаем GPU для FAISS из-за проблем совместимости
        self.gpu_available = False
        
        # Загружаем векторное хранилище
        try:
            if FAISS_AVAILABLE:
                self.vectorstore = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
                print("Индекс FAISS успешно загружен")
            else:
                # Используем простую замену FAISS
                print("Используем простое векторное хранилище вместо FAISS")
                self.vectorstore = SimpleVectorStore.load_local(index_path, self.embeddings)
        except Exception as e:
            print(f"Ошибка при загрузке индекса: {e}")
            import traceback
            traceback.print_exc()
            # Создаем пустое хранилище в случае ошибки
            if FAISS_AVAILABLE:
                self.vectorstore = FAISS.from_texts(["Заглушка для отладки"], self.embeddings)
            else:
                self.vectorstore = SimpleVectorStore(self.embeddings, [Document(page_content="Заглушка для отладки", metadata={"author": "система", "date": "сегодня"})])
        
        # Инициализируем LLM через фабрику
        self.llm = LLMFactory.get_model(model_id, **model_kwargs)
        
        if not self.llm:
            # Если не удалось инициализировать запрошенную модель, используем Phi как запасной вариант
            print(f"Не удалось инициализировать модель '{model_id}', использую Phi как запасной вариант")
            self.llm = LLMFactory.get_model("phi")
        
        self.parser = PydanticOutputParser(pydantic_object=RAGResponse)
        
        print(f"Модель {self.llm.model_name} инициализирована, устройство: {self.llm.device}")
        
        # Информация о CUDA для модели 
        try:
            import torch
            if torch.cuda.is_available() and use_gpu:
                print(f"LLM использует GPU: {torch.cuda.get_device_name(0)}")
                print(f"Доступная память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} ГБ")
                print(f"Используемая память GPU: {torch.cuda.memory_allocated(0) / 1024**3:.2f} ГБ")
        except Exception as e:
            print(f"Ошибка при проверке памяти GPU: {e}")

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Получение релевантных документов."""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.metadata for doc in docs]

    def generate_qa_response(self, query: str) -> str:
        """Генерирует ответ на вопрос с использованием RAG."""
        # Отладочная информация
        print(f"Обработка запроса: {query}")
        
        try:
            # Получаем больше релевантных документов
            docs = self.vectorstore.similarity_search(query, k=15)
            
            if not docs or len(docs) == 0:
                return self._format_error_response("Не найдено релевантной информации по вашему запросу")
            
            # Фильтрация по релевантности к теме
            filtered_docs = []
            for doc in docs:
                text = doc.page_content.lower()
                if any(keyword in text for keyword in CRYPTO_KEYWORDS) or 'vitalik' in doc.metadata['author'].lower():
                    filtered_docs.append(doc)
            
            # Если после фильтрации осталось мало документов, вернем исходные
            if len(filtered_docs) < 3:
                filtered_docs = docs[:15]  # Возьмем хотя бы первые 15
                print("Мало релевантных документов о криптовалютах после фильтрации, используем общие результаты")
            else:
                print(f"Отфильтровано {len(filtered_docs)} релевантных документов о криптовалютах")
            
            # Группируем по авторам для обеспечения разнообразия
            authors_map = {}
            for doc in filtered_docs:
                author = doc.metadata["author"]
                if author not in authors_map:
                    authors_map[author] = []
                authors_map[author].append(doc)

            # Берем макс. 3 документа от каждого автора для обеспечения разнообразия
            balanced_docs = []
            for author, author_docs in authors_map.items():
                balanced_docs.extend(author_docs[:3])
            
            # Если после балансировки осталось мало документов, дополним из общего пула
            if len(balanced_docs) < 3 and len(filtered_docs) >= 3:
                balanced_docs = filtered_docs[:15]  # Используем все документы
            
            # Используем сбалансированные документы, если их достаточно
            final_docs = balanced_docs if balanced_docs else filtered_docs[:15]
            
            # Отладочная информация
            print(f"Использую {len(final_docs)} документов от {len(authors_map)} авторов")
            
            # Форматируем контекст
            formatted_context = []
            for doc in final_docs:
                author = doc.metadata["author"]
                date = doc.metadata["date"]
                formatted_context.append(f"Автор: {author}\nДата: {date}\nТекст: {doc.page_content}\n")
            
            context_text = "\n---\n".join(formatted_context)
            
            # Оптимизированный промпт для языковой модели
            template = """
            Ты - опытный аналитик криптовалютного рынка с глубоким пониманием технологий блокчейн. Твоя задача - ответить на вопрос пользователя, основываясь ТОЛЬКО на предоставленной информации из источников.
            
            ВОПРОС: {question}
            
            ИНФОРМАЦИЯ ИЗ ИСТОЧНИКОВ:
            {context}
            
            Внимательно проанализируй предоставленную информацию и ответь на вопрос пользователя. В своем ответе:
            1. Используй ТОЛЬКО факты из предоставленных текстов, не добавляй внешней информации или своих знаний
            2. Если в текстах нет ответа на вопрос, честно сообщи об этом
            3. Если информация противоречива, укажи на разные точки зрения
            4. Укажи авторов и даты предоставленной информации
            5. Если вопрос не относится к криптовалютам или блокчейну, вежливо объясни, что можешь отвечать только на вопросы по этой тематике
            
            СТРУКТУРА ОТВЕТА (следуй ей строго):
            
            КРАТКИЙ ОТВЕТ: (2-3 предложения, суммирующих ключевую информацию)
            
            ОСНОВНЫЕ ТЕМЫ И ФАКТЫ:
            • Факт 1 из источников
            • Факт 2 из источников
            • Факт 3 из источников (и т.д.)
            
            ЗАКЛЮЧЕНИЕ: (3-4 предложения, обобщающие информацию и представляющие целостную картину по вопросу)
            
            ИСТОЧНИКИ:
            • Автор 1 (дата)
            • Автор 2 (дата)
            • Автор 3 (дата)
            """
            
            prompt = PromptTemplate(
                input_variables=["question", "context"],
                template=template
            )

            # Создаем цепочку для генерации ответа
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt
            )
            
            # Отладочная информация
            print(f"Отправляем запрос к модели {self.llm.model_name}: {query}")
            
            # Получаем ответ
            raw_response = chain.run(
                question=query,
                context=context_text
            )
            
            # Отладочная информация
            print(f"Получен ответ от модели длиной {len(raw_response)} символов")
            
            return self._format_structured_response(raw_response, final_docs)
            
        except Exception as e:
            print(f"Произошла ошибка при генерации ответа: {e}")
            # Более подробная информация об ошибке
            import traceback
            traceback.print_exc()
            return self._format_error_response("Произошла ошибка при обработке запроса. Пожалуйста, попробуйте еще раз.")

    def compare_opinions(self, query: str) -> str:
        """Сравнительный анализ мнений разных авторов."""
        # Отладочная информация
        print(f"Обработка запроса на сравнение мнений: {query}")
        
        try:
            # Получаем больше релевантных документов
            docs = self.vectorstore.similarity_search(query, k=20)
            
            if not docs or len(docs) == 0:
                return self._format_error_response("Не найдено релевантной информации для сравнительного анализа")
            
            # Фильтрация по релевантности к теме
            filtered_docs = []
            for doc in docs:
                text = doc.page_content.lower()
                if any(keyword in text for keyword in CRYPTO_KEYWORDS) or 'vitalik' in doc.metadata['author'].lower():
                    filtered_docs.append(doc)
            
            # Если после фильтрации осталось мало документов, вернем исходные
            if len(filtered_docs) < 3:
                filtered_docs = docs[:15]  # Возьмем хотя бы первые 15
                print("Мало релевантных документов о криптовалютах после фильтрации, используем общие результаты")
            else:
                print(f"Отфильтровано {len(filtered_docs)} релевантных документов о криптовалютах")
            
            # Группируем по авторам
            authors_map = {}
            for doc in filtered_docs:
                author = doc.metadata["author"]
                if author not in authors_map:
                    authors_map[author] = []
                authors_map[author].append(doc)

            # Проверяем, что у нас есть хотя бы 2 автора для сравнения
            if len(authors_map) < 2:
                print(f"Найдено только {len(authors_map)} авторов, недостаточно для сравнения")
                return self._format_error_response("Недостаточно авторов для сравнительного анализа. Найдено только мнение " + 
                                                  list(authors_map.keys())[0] if authors_map else "неизвестного автора")

            # Отбираем наиболее релевантные документы от каждого автора (не более 3)
            for author in authors_map:
                if len(authors_map[author]) > 3:
                    authors_map[author] = authors_map[author][:3]

            # Отладочная информация
            print(f"Авторы для сравнения ({len(authors_map)}): {', '.join(authors_map.keys())}")
            for author, docs_list in authors_map.items():
                print(f"- {author}: {len(docs_list)} документов")

            # Формируем промпт для анализа
            prompt_parts = []
            for author, texts in authors_map.items():
                # Объединяем тексты с метками дат для лучшего контекста
                author_texts = []
                for doc in texts:
                    author_texts.append(f"[{doc.metadata['date']}] {doc.page_content}")
                
                combined_text = "\n".join(author_texts)
                prompt_parts.append(f"Автор: {author}\nТексты: {combined_text}\n")

            # Создаем промпт для анализа, оптимизированный для работы с языковой моделью
            template = """
            Ты - опытный аналитик криптовалютного рынка с глубоким пониманием технологий блокчейн. Твоя задача - выполнить тщательный сравнительный анализ мнений разных экспертов по заданной теме.
            
            ТЕМА ДЛЯ АНАЛИЗА: {query}
            
            МНЕНИЯ ЭКСПЕРТОВ:
            {opinions}
            
            Проведи глубокий анализ, который должен:
            1. Использовать ТОЛЬКО информацию из предоставленных текстов, без внешних знаний
            2. Выделить ключевые позиции каждого автора с прямыми цитатами по возможности
            3. Найти как точки согласия, так и ключевые расхождения во мнениях
            4. Сформировать объективное заключение, основанное на анализе всех мнений
            5. Если в текстах нет явных мнений по заданной теме, честно указать на это
            
            СТРУКТУРА ОТВЕТА (следуй ей строго):
            
            ОСНОВНОЙ ВЫВОД: (2-3 предложения, резюмирующих основные тенденции в мнениях экспертов)
            
            ПОЗИЦИИ АВТОРОВ:
            • [Имя автора 1]: (точное изложение его ключевых аргументов с датами публикаций)
            • [Имя автора 2]: (точное изложение его ключевых аргументов с датами публикаций)
            
            КЛЮЧЕВЫЕ РАСХОЖДЕНИЯ:
            • Расхождение 1: (Автор X считает ____, в то время как Автор Y утверждает ____)
            • Расхождение 2: (если применимо)
            
            ОБЩИЕ ТЕНДЕНЦИИ:
            • Тенденция 1: (точки согласия между авторами)
            • Тенденция 2: (если применимо)
            
            ЗАКЛЮЧЕНИЕ: (3-4 предложения, обобщающие анализ и раскрывающие более глубокое понимание темы)
            
            ИСТОЧНИКИ:
            • [Имя автора 1]
            • [Имя автора 2]
            """

            prompt = PromptTemplate(
                input_variables=["query", "opinions"],
                template=template
            )

            # Создаем цепочку для анализа
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt
            )

            # Отладочная информация
            print(f"Отправляем запрос к модели {self.llm.model_name} для сравнения мнений по теме: {query}")
            
            # Получаем ответ
            raw_response = chain.run(
                query=query,
                opinions="\n".join(prompt_parts)
            )
            
            # Отладочная информация
            print(f"Получен ответ от модели длиной {len(raw_response)} символов")
            
            # Форматируем ответ (эмодзи и структура)
            all_docs = []
            for author_docs in authors_map.values():
                all_docs.extend(author_docs)
            
            return self._format_structured_response(raw_response, all_docs)
        except Exception as e:
            print(f"Произошла ошибка при сравнительном анализе: {e}")
            # Более подробная информация об ошибке
            import traceback
            traceback.print_exc()
            return self._format_error_response("Произошла ошибка при обработке запроса. Пожалуйста, попробуйте еще раз.")

    def _format_structured_response(self, raw_response: str, docs: List[Document]) -> str:
        """Форматирует структурированный ответ с эмодзи и метками источников."""
        # Удаляем возможные служебные инструкции и вступительный текст
        cleaned_response = raw_response
        
        # Удаляем начальные фразы, которые могут оставаться от шаблона
        service_prefixes = [
            "Вот анализ по заданной теме:",
            "Вот ответ на ваш вопрос:",
            "Вот информация по вашему запросу:",
            "Вот мой ответ:",
            "Вот результат анализа:",
            "На основе предоставленных данных,"
        ]
        
        for prefix in service_prefixes:
            if cleaned_response.startswith(prefix):
                cleaned_response = cleaned_response[len(prefix):].strip()
        
        # Проверяем, содержит ли ответ основные секции
        sections = [
            "ОСНОВНОЙ ВЫВОД:", 
            "ПОЗИЦИИ АВТОРОВ:", 
            "КЛЮЧЕВЫЕ РАСХОЖДЕНИЯ:", 
            "ОБЩИЕ ТЕНДЕНЦИИ:", 
            "ЗАКЛЮЧЕНИЕ:", 
            "ИСТОЧНИКИ:"
        ]
        
        # Добавляем соответствующие эмодзи к секциям
        emoji_map = {
            "ОСНОВНОЙ ВЫВОД:": "📌 ",
            "ПОЗИЦИИ АВТОРОВ:": "👤 ",
            "КЛЮЧЕВЫЕ РАСХОЖДЕНИЯ:": "⚔️ ",
            "ОБЩИЕ ТЕНДЕНЦИИ:": "🔄 ",
            "ЗАКЛЮЧЕНИЕ:": "🎯 ",
            "ИСТОЧНИКИ:": "📚 "
        }
        
        # Проверяем наличие секций и добавляем эмодзи
        for section in sections:
            if section in cleaned_response:
                cleaned_response = cleaned_response.replace(section, emoji_map.get(section, "") + section)
        
        # Форматируем источники
        unique_authors = set()
        for doc in docs:
            if 'author' in doc.metadata:
                unique_authors.add(doc.metadata['author'])
        
        sources_text = "\n\n📚 ИСТОЧНИКИ:\n"
        for author in unique_authors:
            sources_text += f"• {author}\n"
        
        # Проверяем, есть ли уже секция ИСТОЧНИКИ в ответе
        if "📚 ИСТОЧНИКИ:" not in cleaned_response:
            cleaned_response += sources_text
        
        # Удаляем возможные множественные пустые строки
        cleaned_response = re.sub(r'\n{3,}', '\n\n', cleaned_response)
        
        # Проверяем, есть ли заключение, если нет - добавляем
        if "🎯 ЗАКЛЮЧЕНИЕ:" not in cleaned_response:
            # Находим последнюю секцию перед источниками
            last_section_pos = -1
            for section in ["ОБЩИЕ ТЕНДЕНЦИИ:", "КЛЮЧЕВЫЕ РАСХОЖДЕНИЯ:", "ПОЗИЦИИ АВТОРОВ:"]:
                section_pos = cleaned_response.find(emoji_map.get(section, "") + section)
                if section_pos > last_section_pos:
                    last_section_pos = section_pos
            
            # Если нашли позицию, добавляем заключение перед источниками
            if last_section_pos != -1:
                sources_pos = cleaned_response.find("📚 ИСТОЧНИКИ:")
                if sources_pos != -1:
                    conclusion = "\n\n🎯 ЗАКЛЮЧЕНИЕ:\nНа основе представленной информации можно сделать вывод, что мнения экспертов по данной теме расходятся. Необходимо учитывать различные точки зрения при формировании собственного мнения.\n"
                    cleaned_response = cleaned_response[:sources_pos] + conclusion + cleaned_response[sources_pos:]
        
        return cleaned_response

    def _format_error_response(self, error_message: str) -> str:
        """Форматирование сообщения об ошибке."""
        return f"""
❌ Ошибка: {error_message}

Пожалуйста, попробуйте:
1. Переформулировать ваш вопрос
2. Использовать другие ключевые слова
3. Задать вопрос по другой теме

Если проблема сохраняется, дайте мне знать, и я помогу вам сформулировать правильный запрос.
"""


def main():
    """Демонстрация RAG-пайплайна."""
    import argparse
    
    # Загружаем переменные окружения
    load_dotenv()
    
    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(description="RAG-система для анализа криптовалютных новостей")
    parser.add_argument("--model", "-m", type=str, default="phi", choices=list(LLMFactory.AVAILABLE_MODELS.keys()),
                        help="Языковая модель для использования (phi, gemma, qwen, saiga)")
    parser.add_argument("--query", "-q", type=str, default="Какие тенденции наблюдаются в Ethereum?",
                        help="Запрос для анализа")
    parser.add_argument("--mode", type=str, default="qa", choices=["qa", "compare"],
                        help="Режим анализа: qa (вопрос-ответ) или compare (сравнение мнений)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                        help="Температура генерации (0.0-1.0)")
    
    # Парсим аргументы
    args = parser.parse_args()
    
    # Отображаем выбранные параметры
    print(f"Модель: {LLMFactory.AVAILABLE_MODELS[args.model]['name']}")
    print(f"Запрос: {args.query}")
    print(f"Режим: {args.mode}")
    print(f"Температура: {args.temperature}")
    print("-" * 50)
    
    # Инициализируем пайплайн с выбранной моделью
    pipeline = RAGPipeline(model_id=args.model, temperature=args.temperature)
    
    # Выполняем запрос в зависимости от режима
    if args.mode == "qa":
        print("\n=== Ответ на вопрос ===")
        answer = pipeline.generate_qa_response(args.query)
        print(answer)
    else:
        print("\n=== Сравнительный анализ ===")
        analysis = pipeline.compare_opinions(args.query)
        print(analysis)


if __name__ == "__main__":
    main() 