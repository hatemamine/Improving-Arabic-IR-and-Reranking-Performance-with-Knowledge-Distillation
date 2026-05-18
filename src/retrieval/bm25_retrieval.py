from __future__ import annotations

import logging
import os
import subprocess
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25 retrieval with two backends:

    1. Pure-Python  (rank_bm25, no Java) — used by build_run()
       Tokenises on whitespace, builds an in-memory BM25 index, writes a
       TREC-format run file.  Suitable for mMARCO / Mr.TyDi Arabic corpora
       that are already pre-tokenised / preprocessed.

    2. Pyserini     (requires Java)      — used by retrieve()
       Wraps pyserini.search.lucene for production-quality Lucene BM25.
    """

    def __init__(
        self,
        language: str = "ar",
        hits: int = 1000,
        threads: int = 16,
        batch_size: int = 128,
    ):
        self.language = language
        self.hits = hits
        self.threads = threads
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Pure-Python backend (rank_bm25, no Java required)
    # ------------------------------------------------------------------

    def build_run(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        output_path: str,
        top_k: int = 1000,
        batch_size: int = 500,
    ) -> Dict[str, List[str]]:
        """
        Build a BM25 run with rank_bm25 (pip install rank-bm25).
        No Java or Pyserini required.

        Args:
            queries:     {qid: query_text}
            corpus:      {doc_id: doc_text}
            output_path: where to write the TREC run file
            top_k:       candidates per query
            batch_size:  queries processed per batch (memory control)

        Returns:
            run dict {qid: [doc_id, ...]} (sorted by score descending)
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 is required: pip install rank-bm25")

        import numpy as np

        logger.info("Building BM25 index over %d documents…", len(corpus))
        doc_ids   = list(corpus.keys())
        doc_texts = [corpus[d].split() for d in doc_ids]
        bm25 = BM25Okapi(doc_texts)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        run: Dict[str, List[str]] = {}
        qid_list = list(queries.keys())

        with open(output_path, "w", encoding="utf-8") as fout:
            for start in range(0, len(qid_list), batch_size):
                batch_qids = qid_list[start : start + batch_size]
                logger.info(
                    "BM25 scoring queries %d–%d / %d",
                    start + 1, min(start + batch_size, len(qid_list)), len(qid_list),
                )
                for qid in batch_qids:
                    tokens = queries[qid].split()
                    scores = bm25.get_scores(tokens)
                    top_idx = np.argpartition(scores, -min(top_k, len(scores)))[
                        -min(top_k, len(scores)):
                    ]
                    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
                    ranked = [str(doc_ids[i]) for i in top_idx]
                    run[str(qid)] = ranked
                    for rank, did in enumerate(ranked, start=1):
                        fout.write(f"{qid}\tQ0\t{did}\t{rank}\t{scores[top_idx[rank-1]]:.6f}\tbm25\n")

        logger.info("BM25 run saved to %s (%d queries)", output_path, len(run))
        return run

    # ------------------------------------------------------------------
    # Pyserini backend (requires Java)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        topics: str,
        index: str,
        output_path: str,
        extra_args: Optional[List[str]] = None,
    ) -> str:
        """
        Run Pyserini BM25 search (requires Java + pyserini).

        Args:
            topics:      Pyserini topic name or path to topics file
            index:       Pyserini index name or path to local index
            output_path: path to write the run file
            extra_args:  additional CLI arguments

        Returns:
            Path to the output run file
        """
        cmd = [
            "python", "-m", "pyserini.search.lucene",
            "--threads", str(self.threads),
            "--batch-size", str(self.batch_size),
            "--language", self.language,
            "--topics", topics,
            "--index", index,
            "--output", output_path,
            "--bm25",
            "--hits", str(self.hits),
        ]
        if extra_args:
            cmd.extend(extra_args)

        logger.info("Running BM25: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"BM25 retrieval failed:\n{result.stderr}")
        logger.info("BM25 run saved to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def load_run(
        self,
        path: str,
        k: int = 1000,
    ) -> Dict[str, List[str]]:
        """Load a TREC-format run file. Returns {qid: [doc_id, ...]}."""
        import pandas as pd

        df = pd.read_csv(path, sep=r"\s+", header=None)
        df.columns = ["qid", "q0", "docid", "rank", "score", "tag"]
        df = df[df["rank"] <= k]
        run = df.groupby("qid")["docid"].apply(list).to_dict()
        return {str(k): [str(v) for v in vs] for k, vs in run.items()}

    @staticmethod
    def evaluate(
        qrels_path: str,
        run_path: str,
        metrics: Optional[List[str]] = None,
    ) -> str:
        """Run trec_eval via pyserini and return stdout."""
        if metrics is None:
            metrics = ["recip_rank.10", "recall.1000", "ndcg_cut.10", "map_cut.10"]

        metric_args = []
        for m in metrics:
            metric_args += ["-m", m]

        cmd = ["python", "-m", "pyserini.eval.trec_eval", "-c"] + metric_args + [qrels_path, run_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"trec_eval failed:\n{result.stderr}")
        return result.stdout

