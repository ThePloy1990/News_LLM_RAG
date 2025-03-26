import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Загрузка переменных окружения
load_dotenv()


def load_data(filepath: str) -> List[Dict]:
    """Загрузка данных из JSON файла."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def create_documents(data: List[Dict]) -> List[Document]:
    """Создание документов для индексации."""
    documents = []
    
    for item in data:
        # Создаем метаданные
        metadata = {
            "source": item["source"],
            "author": item["author"],
            "date": str(item["date"]),
            "id": item["id"]
        }
        
        if "url" in item:
            metadata["url"] = item["url"]
            
        # Создаем документ
        doc = Document(
            page_content=item["text"],
            metadata=metadata
        )
        documents.append(doc)
    
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Разделение документов на чанки."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents(documents)


def build_index(documents: List[Document], output_path: str):
    """Построение и сохранение индекса."""
    # Инициализация эмбеддингов
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Создание векторного хранилища
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Сохранение индекса
    vectorstore.save_local(output_path)
    
    print(f"Индекс сохранен в {output_path}")
    print(f"Количество документов в индексе: {len(documents)}")


def main():
    """Основная функция для построения индекса."""
    print("Начинаем построение индекса...")
    
    # Загрузка данных
    data = load_data("data/all_posts.json")
    print(f"Загружено {len(data)} сообщений")
    
    # Создание документов
    documents = create_documents(data)
    print(f"Создано {len(documents)} документов")
    
    # Разделение на чанки
    split_docs = split_documents(documents)
    print(f"Разделено на {len(split_docs)} чанков")
    
    # Построение индекса
    os.makedirs("faiss_index", exist_ok=True)
    build_index(split_docs, "faiss_index")
    
    print("Индексация завершена")


if __name__ == "__main__":
    main() 