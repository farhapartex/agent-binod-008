import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import Tool

def load_documents():
    documents = []

    sample_docs = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models. It provides tools for document loading, text splitting, and vector storage.",
            metadata={"source": "langchain_intro.txt"}
        ),
        Document(
            page_content="Vector databases store embeddings of text chunks. Popular options include Chroma, Pinecone, and FAISS. They enable semantic search.",
            metadata={"source": "vector_db_info.txt"}
        ),
        Document(
            page_content="OpenAI provides powerful language models like GPT-4 and GPT-3.5-turbo. These models can understand context and generate human-like responses.",
            metadata={"source": "openai_models.txt"}
        )
    ]
    return sample_docs


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


