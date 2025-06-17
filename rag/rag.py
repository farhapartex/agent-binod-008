from typing import Dict, List

from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain_core.stores import InMemoryStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever

class ParentDocumentRAG:
    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.vectorstore = None
        self.retriever = None

    def create_parent_document_retriever(self, documents: List[Document]):
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )

        vectorstore = Chroma.from_documents(
            documents=[],
            embedding=self.embeddings,
            persist_directory="./parent_doc_db"
        )

        store = InMemoryStore()

        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
        retriever.add_documents(documents)
        self.vectorstore = vectorstore
        self.retriever = retriever

    def search(self, query: str) -> List[Document]:
        if not self.retriever:
            raise ValueError("Retriever not initialized")

        results = self.retriever.get_relevant_documents(query)

        # for i, doc in enumerate(results):
        #     print(f"{i + 1}. Content length: {len(doc.page_content)} chars")
        #     print(f"   Metadata: {doc.metadata}")
        #     print(f"   Preview: {doc.page_content[:100]}...")
        #     print("-" * 40)

        return results

class MultiQueryRAG:
    def __init__(self, vectorstore: Chroma, llm: ChatOpenAI):
        self.vectorstore = vectorstore
        self.llm = llm

    def create_multi_query_retriever(self) -> MultiQueryRetriever:
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm
        )

        return multi_query_retriever

    def search(self, query: str) -> List[Document]:
        retriever = self.create_multi_query_retriever()
        results = retriever.get_relevant_documents(query)
        return results


class HybridSearchRAG:
    """Hybrid Search: Combine semantic search with keyword search"""

    def __init__(self, documents: List[Document], embeddings: OpenAIEmbeddings):
        self.documents = documents
        self.embeddings = embeddings
        self.hybrid_retriever = None

    def create_hybrid_retriever(self) -> EnsembleRetriever:
        """Create hybrid retriever combining semantic and keyword search"""
        vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            persist_directory="./hybrid_db"
        )
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        keyword_retriever = BM25Retriever.from_documents(self.documents)
        keyword_retriever.k = 5

        hybrid_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, keyword_retriever],
            weights=[0.6, 0.4]  # 60% semantic, 40% keyword
        )
        self.hybrid_retriever = hybrid_retriever
        return hybrid_retriever

    def search(self, query: str) -> List[Document]:
        """Search using hybrid approach"""
        if not self.hybrid_retriever:
            self.create_hybrid_retriever()

        results = self.hybrid_retriever.get_relevant_documents(query)
        return results