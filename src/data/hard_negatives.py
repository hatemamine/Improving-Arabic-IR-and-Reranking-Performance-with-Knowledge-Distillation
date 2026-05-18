from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """
    Mines hard negatives using BM25 retrieved lists and/or ANCE-style
    bi-encoder dynamic refresh.

    Curriculum schedule selects which strategy is active per epoch:
      epoch 1 → bm25
      epoch 2 → combined (50/50 bm25 + ance)
      epoch 3 → ance
    """

    def __init__(
        self,
        strategy: str = "curriculum",
        bm25_top_k: int = 200,
        ance_top_k: int = 200,
        curriculum_schedule: Optional[List[Tuple[int, str]]] = None,
        seed: int = 42,
    ):
        if strategy not in ("bm25", "ance", "combined", "curriculum"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self.bm25_top_k = bm25_top_k
        self.ance_top_k = ance_top_k
        self.curriculum_schedule = curriculum_schedule or [
            (1, "bm25"),
            (2, "combined"),
            (3, "ance"),
        ]
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Populated externally
        self._bm25_negatives: Dict[str, List[str]] = {}   # qid → [doc_ids]
        self._ance_negatives: Dict[str, List[str]] = {}   # qid → [doc_ids]
        self._qrels: Dict[str, set] = {}                  # qid → {relevant doc_ids}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_bm25_negatives(
        self,
        bm25_run: Dict[str, List[str]],
        qrels: Dict[str, Dict[str, int]],
    ):
        """
        Derive hard negatives from a BM25 run by keeping retrieved docs that
        are NOT in qrels (non-relevant top-k retrieved passages).
        """
        self._qrels = {qid: set(docs.keys()) for qid, docs in qrels.items()}
        for qid, retrieved in bm25_run.items():
            relevant = self._qrels.get(str(qid), set())
            hard = [d for d in retrieved if str(d) not in relevant]
            self._bm25_negatives[str(qid)] = hard[: self.bm25_top_k]
        logger.info("BM25 hard negatives loaded for %d queries", len(self._bm25_negatives))

    def refresh_ance_negatives(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        bi_encoder,
        batch_size: int = 128,
    ):
        """
        Re-build ANCE negatives using current bi-encoder state.
        Encodes all corpus docs and retrieves top-K per query, then
        removes relevant docs to leave hard negatives.
        """
        try:
            import faiss
        except ImportError:
            logger.warning("faiss not installed — skipping ANCE refresh")
            return

        logger.info("Refreshing ANCE negatives with current bi-encoder…")
        doc_ids = list(corpus.keys())
        doc_texts = [corpus[d] for d in doc_ids]

        # Encode corpus
        corpus_embeddings = bi_encoder.encode(
            doc_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        dim = corpus_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_embeddings.astype(np.float32))

        # Encode queries and search
        query_ids = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]
        query_embeddings = bi_encoder.encode(
            query_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        _, indices = index.search(query_embeddings.astype(np.float32), self.ance_top_k + 10)

        for qid, idxs in zip(query_ids, indices):
            relevant = self._qrels.get(str(qid), set())
            hard = [
                str(doc_ids[i])
                for i in idxs
                if str(doc_ids[i]) not in relevant
            ]
            self._ance_negatives[str(qid)] = hard[: self.ance_top_k]

        logger.info("ANCE negatives refreshed for %d queries", len(self._ance_negatives))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def get_negatives(
        self,
        qid: str,
        n: int = 1,
        epoch: int = 1,
    ) -> List[str]:
        """Return n hard negative doc_ids for query qid given current epoch."""
        active = self._active_strategy(epoch)
        pool = self._get_pool(qid, active)
        if not pool:
            return []
        return random.sample(pool, min(n, len(pool)))

    def _active_strategy(self, epoch: int) -> str:
        if self.strategy != "curriculum":
            return self.strategy
        active = "bm25"
        for ep, strat in sorted(self.curriculum_schedule):
            if epoch >= ep:
                active = strat
        return active

    def _get_pool(self, qid: str, strategy: str) -> List[str]:
        qid = str(qid)
        if strategy == "bm25":
            return self._bm25_negatives.get(qid, [])
        if strategy == "ance":
            return self._ance_negatives.get(qid, [])
        if strategy == "combined":
            bm25 = self._bm25_negatives.get(qid, [])
            ance = self._ance_negatives.get(qid, [])
            combined = list(set(bm25 + ance))
            return combined
        return []

    def augment_dataset_with_hard_negatives(
        self,
        dataset,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        epoch: int = 1,
        neg_per_query: int = 1,
    ):
        """
        Returns a new list of (query_text, pos_text, neg_text, label) tuples
        with hard negatives substituted where available.
        """
        rows = []
        for sample in dataset:
            qid = str(sample.get("query", sample.get("qid", "")))
            pos_id = str(sample.get("pos", sample.get("pos_id", "")))
            label = sample.get("label", 0.0)

            hard_neg_ids = self.get_negatives(qid, n=neg_per_query, epoch=epoch)
            for neg_id in hard_neg_ids:
                if qid in queries and pos_id in corpus and neg_id in corpus:
                    rows.append({
                        "query": queries[qid],
                        "positive": corpus[pos_id],
                        "negative": corpus[neg_id],
                        "label": float(label),
                    })
        return rows
