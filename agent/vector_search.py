import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import Tool

def load_documents(pdf_file_path):
    documents = []
    pdf_files = [
        pdf_file_path
    ]

    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            try:
                loader = PyPDFLoader(pdf_file)
                pdf_docs = loader.load()

                for i, doc in enumerate(pdf_docs):
                    doc.metadata.update({
                        "source": pdf_file,
                        "page": i+1,
                        "file_type": "pdf"
                    })

                documents.extend(pdf_docs)
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


