# FIXED powers.py - Add validation before creating vector stores

from typing import Tuple, List, Dict, Any
from datetime import datetime, timedelta
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain_core.stores import InMemoryStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import Tool
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from sentence_transformers import CrossEncoder

from rag.processor import AdvanceDocumentProcessor


class ParentDocumentRAG:
    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.vectorstore = None
        self.retriever = None

    def create_parent_document_retriever(self, documents: List[Document]):
        valid_docs = self._validate_documents(documents)
        if not valid_docs:
            return None

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

        try:
            retriever.add_documents(valid_docs)
            self.vectorstore = vectorstore
            self.retriever = retriever
        except Exception as e:
            return None

    def _validate_documents(self, documents: List[Document]) -> List[Document]:
        valid_docs = []
        for i, doc in enumerate(documents):
            if doc.page_content and len(doc.page_content.strip()) > 50:
                # Ensure metadata is safe
                safe_metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        safe_metadata[key] = value
                    else:
                        safe_metadata[key] = str(value)
                doc.metadata = safe_metadata
                valid_docs.append(doc)
        return valid_docs

    def search(self, query: str) -> List[Document]:
        if not self.retriever:
            return []

        try:
            results = self.retriever.get_relevant_documents(query)
            return results
        except Exception as e:
            return []


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
        try:
            retriever = self.create_multi_query_retriever()
            results = retriever.get_relevant_documents(query)
            return results
        except Exception as e:
            return []


class HybridSearchRAG:
    def __init__(self, documents: List[Document], embeddings: OpenAIEmbeddings):
        self.documents = self._validate_documents(documents)  # FIXED: Validate first
        self.embeddings = embeddings
        self.hybrid_retriever = None

    def _validate_documents(self, documents: List[Document]) -> List[Document]:
        valid_docs = []
        for doc in documents:
            if doc.page_content and len(doc.page_content.strip()) > 20:
                # Ensure clean metadata
                safe_metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        safe_metadata[key] = value
                    else:
                        safe_metadata[key] = str(value)
                doc.metadata = safe_metadata
                valid_docs.append(doc)
        return valid_docs

    def create_hybrid_retriever(self) -> EnsembleRetriever:
        if not self.documents:
            return None

        try:
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

        except Exception as e:
            return None

    def search(self, query: str) -> List[Document]:
        if not self.hybrid_retriever:
            result = self.create_hybrid_retriever()
            if not result:
                return []

        try:
            results = self.hybrid_retriever.get_relevant_documents(query)
            return results
        except Exception as e:
            return []


class ReRankingRAG:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            self.cross_encoder = None

    def search(
            self,
            query: str,
            top_k: int = 3,
            initial_k: int = 10
    ) -> List[Tuple[Document, float]]:

        if not self.cross_encoder:
            try:
                docs = self.vectorstore.similarity_search(query, k=top_k)
                return [(doc, 1.0) for doc in docs]  # Return with dummy scores
            except Exception as e:
                return []

        try:
            initial_docs = self.vectorstore.similarity_search(query, k=initial_k)

            if not initial_docs:
                return []

            query_doc_pairs = [(query, doc.page_content) for doc in initial_docs]
            scores = self.cross_encoder.predict(query_doc_pairs)
            doc_score_pairs = list(zip(initial_docs, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            top_results = doc_score_pairs[:top_k]
            return top_results

        except Exception as e:
            return []


class MetadataFilterRAG:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def search_with_filters(self, query: str, filters: Dict[str, Any]) -> List[Document]:
        try:
            where_clause = self._build_where_clause(filters)
            results = self.vectorstore.similarity_search(
                query,
                k=5,
                filter=where_clause
            )
            return results
        except Exception as e:
            return []

    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict:
        where_clause = {}

        for key, value in filters.items():
            if isinstance(value, list):
                where_clause[key] = {"$in": value}
            elif isinstance(value, dict):
                where_clause[key] = value
            else:
                where_clause[key] = {"$eq": value}

        return where_clause

    def search_by_content_type(self, query: str, content_type: str) -> List[Document]:
        return self.search_with_filters(query, {"content_type": content_type})

    def search_by_page_range(self, query: str, min_page: int, max_page: int) -> List[Document]:
        return self.search_with_filters(query, {
            "page": {"$gte": min_page, "$lte": max_page}
        })

    def search_recent_content(self, query: str, days_back: int = 30) -> List[Document]:
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        return self.search_with_filters(query, {
            "modified_date": {"$gte": cutoff_date}
        })


class AdvancedRAGSystem:
    def __init__(self, llm: ChatOpenAI, vectorstore=None):
        self.llm = llm
        self.embeddings = OpenAIEmbeddings()
        self.doc_processor = AdvanceDocumentProcessor()

        # Storage for different retrieval methods
        self.vectorstore = vectorstore
        self.parent_doc_rag = None
        self.multi_query_rag = None
        self.hybrid_rag = None
        self.reranking_rag = None
        self.metadata_filter_rag = None

    def _setup_vectorstore(self, docs):
        try:
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory="./advanced_rag_db"
            )
        except Exception as e:
            return False

    def setup_all_retrievers(self, pdf_files: List[str]):
        # Load documents
        documents = self.doc_processor.load_documents(pdf_files)
        if not documents:
            return False

        valid_docs = self._validate_documents_for_vectorstore(documents)
        if not valid_docs:
            return False

        if not self.vectorstore:
            self._setup_vectorstore(valid_docs)

        try:
            self.parent_doc_rag = ParentDocumentRAG(self.embeddings)
            self.parent_doc_rag.create_parent_document_retriever(valid_docs)
        except Exception as e:
            pass

        try:
            # Setup Multi-Query Retrieval
            self.multi_query_rag = MultiQueryRAG(self.vectorstore, self.llm)
        except Exception as e:
            pass

        try:
            # Setup Hybrid Search
            self.hybrid_rag = HybridSearchRAG(valid_docs, self.embeddings)
            self.hybrid_rag.create_hybrid_retriever()
        except Exception as e:
            pass

        try:
            # Setup Re-ranking
            self.reranking_rag = ReRankingRAG(self.vectorstore)
        except Exception as e:
            pass

        try:
            # Setup Metadata Filtering
            self.metadata_filter_rag = MetadataFilterRAG(self.vectorstore)
        except Exception as e:
            pass

        return True

    def _validate_documents_for_vectorstore(self, documents: List[Document]) -> List[Document]:
        valid_docs = []

        for i, doc in enumerate(documents):
            if not doc.page_content or len(doc.page_content.strip()) < 20:
                continue

            # Validate metadata
            safe_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    safe_metadata[key] = value
                elif isinstance(value, list):
                    safe_metadata[key] = ", ".join(str(item) for item in value)
                else:
                    safe_metadata[key] = str(value)

            doc.metadata = safe_metadata
            valid_docs.append(doc)

        return valid_docs

    def search_with_strategy(self, query: str, strategy: str = "basic") -> List[Document]:
        strategies = {
            "basic": lambda q: self.vectorstore.similarity_search(q,
                                                                  k=3) if self.vectorstore else [],
            "parent_doc": lambda q: self.parent_doc_rag.search(q) if self.parent_doc_rag else [],
            "multi_query": lambda q: self.multi_query_rag.search(q) if self.multi_query_rag else [],
            "hybrid": lambda q: self.hybrid_rag.search(q) if self.hybrid_rag else [],
            "reranked": lambda q: [doc for doc, score in
                                   self.reranking_rag.search(q)] if self.reranking_rag else [],
        }

        if strategy not in strategies:
            return []

        try:
            return strategies[strategy](query)
        except Exception as e:
            return []

    def create_advanced_rag_tool(self, strategy: str = "basic") -> Tool:
        def advanced_search(query: str) -> str:
            try:
                if not query or not query.strip():
                    return "Please provide a valid search query."

                docs = self.search_with_strategy(query, strategy)

                if not docs:
                    return "NO_RELEVANT_DOCS_FOUND"

                result = f"Found using {strategy} search:\n\n"

                for i, doc in enumerate(docs[:3], 1):
                    content = doc.page_content[:300]
                    if len(doc.page_content) > 300:
                        content += "..."

                    result += f"{i}. {content}\n"

                    metadata = doc.metadata
                    source_info = f"Source: {metadata.get('filename', 'Unknown')}"
                    if metadata.get('page'):
                        source_info += f", Page {metadata['page']}"
                    if metadata.get('content_type'):
                        source_info += f", Type: {metadata['content_type']}"

                    result += f"   {source_info}\n\n"

                return result.strip()

            except Exception as e:
                return f"NO_RELEVANT_DOCS_FOUND - Error: {str(e)[:100]}"

        return Tool(
            name="advanced_rag_search",
            description=f"Advanced RAG search using {strategy} strategy. Searches through documents with enhanced relevance and metadata.",
            func=advanced_search
        )
