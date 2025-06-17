import os
import datetime
from typing import Dict, List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import Tool

def _extract_document_metadata(file_path: str, doc: List[Document]) -> Dict:
    file_stats = os.stat(file_path)
    metadata = {
        "filename": os.path.basename(file_path),
        "file_size": file_stats.st_size,
        "created_date": datetime.datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
        "modified_date": datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
        "total_page": len(doc),
        "estimated_read_time": len(doc) * 2
    }

    if doc:
        first_page = doc[0].page_content[:500]
        lines = first_page.split('\n')
        potential_title = next((line.strip() for line in lines if len(line.strip()) > 10), "Unknown")
        metadata["extracted_title"] = potential_title

    return metadata

def _categorize_content(content: str) -> str:
    content_lower = content.lower()

    if any(word in content_lower for word in ['chapter', 'introduction', 'conclusion']):
        return "chapter"
    elif any(word in content_lower for word in ['table', 'figure', 'chart', 'graph']):
        return "data_visualization"
    elif any(word in content_lower for word in ['references', 'bibliography', 'citations']):
        return "references"
    elif len(content.split()) < 100:
        return "short_content"
    else:
        return "main_content"

def _extract_key_terms(content: str, max_terms: int = 5) -> List[str]:
    words = content.lower().split()
    meaningful_words = [word for word in words
                        if len(word) > 3 and word.isalpha()]
    word_freq = {}
    for word in meaningful_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [term[0] for term in sorted_terms[:max_terms]]

def _enhance_page_metadata(doc: Document, page_num: int, doc_metadata: Dict, file_path: str) -> Dict:
    content = doc.page_content
    enhanced = {
        "source": file_path,
        "page": page_num + 1,
        "file_type": "pdf",
        "character_count": len(content),
        "word_count": len(content.split()),
        "has_numbers": any(char.isdigit() for char in content),
        "has_bullet_points": any(marker in content for marker in ['•', '○', '-', '1.', '2.']),
        "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
    }
    enhanced.update(doc_metadata)
    enhanced["content_type"] = _categorize_content(content)
    enhanced["key_terms"] = _extract_key_terms(content)

    return enhanced

def load_documents(pdf_file_path):
    documents = []
    pdf_files = [
        pdf_file_path
    ]

    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file):
            continue
        try:
            loader = PyPDFLoader(pdf_file)
            pdf_docs = loader.load()
            doc_metadata = _extract_document_metadata(pdf_file, pdf_docs)

            for i, doc in enumerate(pdf_docs):
                if doc.page_content.strip():
                    enhance_metadata = _enhance_page_metadata(doc, i, doc_metadata, pdf_file)
                    doc.metadata.update(enhance_metadata)

                documents.append(doc)
        except Exception as e:
            print(f"Error loading pdf: {e}")

    return documents


def split_documents(documents):
    """Split Documents into Chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    splits = text_splitter.split_documents(documents)
    return splits


def create_vector_store(documents) -> Chroma:
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

def get_rag_tool(vectorstore: Chroma):
    def search_documents(query: str) -> str:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant data found"

        content = "\n\n".join([doc.page_content for doc in docs])
        return content

    return Tool(
        name="vector_search",
        description="Search through uploaded documents for relevant information",
        func=search_documents
    )

