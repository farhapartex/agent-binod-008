from rag.powers import AdvancedRAGSystem
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def demonstrate_advanced_rag():
    pdf_files = [
        "/home/ubuntu/Downloads/Mindset _ The New Psychology of Success ( PDFDrive ).pdf"
    ]

    llm = ChatOpenAI(temperature=0)
    advanced_rag = AdvancedRAGSystem(llm)
    success = advanced_rag.setup_all_retrievers(pdf_files)

    if not success:
        return

    # Test queries
    test_queries = [
        "What is a growth mindset?",
        "How does praise affect children?",
        "What are the characteristics of fixed mindset?",
    ]

    # Test available strategies
    strategies = ["basic"]  # Start with basic, add others if they work
    if advanced_rag.parent_doc_rag and advanced_rag.parent_doc_rag.retriever:
        strategies.append("parent_doc")
    if advanced_rag.multi_query_rag:
        strategies.append("multi_query")
    if advanced_rag.hybrid_rag and advanced_rag.hybrid_rag.hybrid_retriever:
        strategies.append("hybrid")
    if advanced_rag.reranking_rag:
        strategies.append("reranked")

    for query in test_queries:
        print(f"\n" + "=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        for strategy in strategies:
            try:
                print(f"\n--- {strategy.upper()} STRATEGY ---")
                results = advanced_rag.search_with_strategy(query, strategy)
                print(f"Results count: {len(results)}")
                if results:
                    print(f"First result preview: {results[0].page_content[:100]}...")
            except Exception as e:
                print(f"{strategy} failed: {e}")

    # Create and test advanced tool
    try:
        print(f"\nTesting Advanced RAG Tool:")
        advanced_tool = advanced_rag.create_advanced_rag_tool("basic")  # Use basic strategy
        result = advanced_tool.func("What is growth mindset?")
        print(result[:500] + "..." if len(result) > 500 else result)
    except Exception as e:
        print(f"Tool testing failed: {e}")



demonstrate_advanced_rag()