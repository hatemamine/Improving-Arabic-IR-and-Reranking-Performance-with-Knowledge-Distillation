from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FaissRetriever:
    """
    Dense first-stage retrieval using FAISS.
    Encodes the corpus once, builds an IP index, then retrieves top-K
    for each query.
    """

    def __init__(
        self,
        bi_encoder,
        top_k: int = 1000,
        batch_size: int = 128,
        use_gpu: bool = False,
        index_path: Optional[str] = None,
    ):
        self.bi_encoder = bi_encoder
        self.top_k = top_k
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.index_path = index_path
        self._index = None
        self._doc_ids: List[str] = []

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, corpus: Dict[str, str], save_path: Optional[str] = None):
        """Encode corpus and build FAISS IP index."""
        import faiss

        self._doc_ids = list(corpus.keys())
        doc_texts = [corpus[d] for d in self._doc_ids]

        logger.info("Encoding %d corpus documents…", len(doc_texts))
        embeddings = self.bi_encoder.encode(
            doc_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(embeddings)
        self._index = index
        logger.info("FAISS index built: %d vectors (dim=%d)", index.ntotal, dim)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            faiss.write_index(faiss.index_gpu_to_cpu(index) if self.use_gpu else index, save_path)
            logger.info("Index saved to %s", save_path)

    def load_index(self, path: str, doc_ids: List[str]):
        import faiss
        self._index = faiss.read_index(path)
        self._doc_ids = doc_ids
        logger.info("Loaded FAISS index from %s (%d vectors)", path, self._index.ntotal)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        queries: Dict[str, str],
        top_k: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """Return {qid: [doc_id, ...]} ranked by score, highest first."""
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        k = top_k or self.top_k
        query_ids = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]

        logger.info("Encoding %d queries…", len(query_texts))
        q_embeddings = self.bi_encoder.encode(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        _, indices = self._index.search(q_embeddings, k)

        results: Dict[str, List[str]] = {}
        for qid, idxs in zip(query_ids, indices):
            results[str(qid)] = [str(self._doc_ids[i]) for i in idxs if i >= 0]
        return results

    def save_run(
        self,
        results: Dict[str, List[str]],
        output_path: str,
        run_tag: str = "faiss",
    ):
        """Save retrieval results in TREC run format."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for qid, doc_ids in results.items():
                for rank, doc_id in enumerate(doc_ids, start=1):
                    f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{-rank}\t{run_tag}\n")
        logger.info("Run saved to %s", output_path)
