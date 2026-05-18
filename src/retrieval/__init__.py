def __getattr__(name):
    if name == "FaissRetriever":
        from src.retrieval.faiss_retrieval import FaissRetriever
        return FaissRetriever
    if name == "BM25Retriever":
        from src.retrieval.bm25_retrieval import BM25Retriever
        return BM25Retriever
    raise AttributeError(f"module 'src.retrieval' has no attribute {name!r}")
