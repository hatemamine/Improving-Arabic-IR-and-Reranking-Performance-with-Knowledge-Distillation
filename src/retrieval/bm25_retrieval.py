from __future__ import annotations

import logging
import os
import subprocess
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Thin wrapper around Pyserini BM25 retrieval for Arabic.
    Assumes pyserini is installed and Java is available.
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

    def retrieve(
        self,
        topics: str,
        index: str,
        output_path: str,
        extra_args: Optional[List[str]] = None,
    ) -> str:
        """
        Run Pyserini BM25 search.

        Args:
            topics:      Pyserini topic name (e.g. "mrtydi-v1.1-arabic-test") or path to topics file
            index:       Pyserini index name (e.g. "mrtydi-v1.1-ar") or path to local index
            output_path: Path to write the run file
            extra_args:  Additional CLI arguments to pass to pyserini

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

    def load_run(
        self,
        path: str,
        k: int = 1000,
    ) -> Dict[str, List[str]]:
        """Load a TREC-format run file. Returns {qid: [doc_id, ...]}."""
        import pandas as pd

        df = pd.read_csv(path, sep=r"\s+", header=None)
        df = df[df[2] <= k]
        run = df.groupby(0)[1].apply(list).to_dict()
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
