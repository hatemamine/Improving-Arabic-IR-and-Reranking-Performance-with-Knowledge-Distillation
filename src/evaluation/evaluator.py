from __future__ import annotations

import logging
import operator
import os
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder

from src.config.config import EvalConfig, PipelineConfig
from src.evaluation.metrics import MetricsResult, compute_metrics

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """
    End-to-end evaluator.

    Usage:
        evaluator = PipelineEvaluator(config, queries, corpus, qrels)
        result = evaluator.evaluate_first_stage(bi_encoder_model, run)
        result = evaluator.evaluate_reranking(cross_encoder_model, first_stage_run)
    """

    def __init__(
        self,
        config: PipelineConfig,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
    ):
        self.config = config
        self.queries = queries
        self.corpus = corpus
        self.qrels = qrels
        self.eval_cfg: EvalConfig = config.evaluation

    # ------------------------------------------------------------------
    # First-stage evaluation
    # ------------------------------------------------------------------

    def evaluate_first_stage(
        self,
        run: Dict[str, List[str]],
        model_name: str = "bi_encoder",
        kd_mode: str = "nokd",
    ) -> MetricsResult:
        k_values = self._parse_k_values()
        scores = compute_metrics(run, self.qrels, k_values)
        return self._build_result(scores, model_name, "first_stage", kd_mode)

    # ------------------------------------------------------------------
    # Reranking evaluation
    # ------------------------------------------------------------------

    def evaluate_reranking(
        self,
        reranker,
        first_stage_run: Dict[str, List[str]],
        model_name: str = "cross_encoder",
        kd_mode: str = "nokd",
        use_cross_encoder: bool = True,
        top_k: Optional[int] = None,
    ) -> MetricsResult:
        k = top_k or self.eval_cfg.rerank_top_k
        reranked_run = self._rerank(reranker, first_stage_run, k, use_cross_encoder)
        k_values = self._parse_k_values()
        scores = compute_metrics(reranked_run, self.qrels, k_values)
        return self._build_result(scores, model_name, "reranking", kd_mode)

    # ------------------------------------------------------------------
    # Reranking helpers
    # ------------------------------------------------------------------

    def _rerank(
        self,
        reranker,
        first_stage_run: Dict[str, List[str]],
        top_k: int,
        use_cross_encoder: bool,
    ) -> Dict[str, List[str]]:
        reranked: Dict[str, List[str]] = {}
        items = list(first_stage_run.items())

        for qid, doc_ids in items:
            query_text = self.queries.get(str(qid), "")
            candidates = doc_ids[:top_k]

            if use_cross_encoder:
                pairs = [[query_text, self.corpus.get(str(d), "")] for d in candidates]
                with torch.no_grad():
                    scores = reranker.predict(pairs, batch_size=self.eval_cfg.eval_batch_size)
            else:
                # Bi-encoder reranking
                with torch.no_grad():
                    q_emb = reranker.encode(query_text, convert_to_tensor=True)
                    doc_texts = [self.corpus.get(str(d), "") for d in candidates]
                    d_embs = reranker.encode(
                        doc_texts,
                        convert_to_tensor=True,
                        batch_size=self.eval_cfg.eval_batch_size,
                    )
                    scores = [util.cos_sim(q_emb, d)[0].item() for d in d_embs]

            scored = sorted(zip(candidates, scores), key=operator.itemgetter(1), reverse=True)
            reranked[str(qid)] = [str(d) for d, _ in scored]

        return reranked

    def save_run(
        self,
        run: Dict[str, List[str]],
        output_path: str,
        run_tag: str = "run",
    ):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for qid, doc_ids in run.items():
                for rank, doc_id in enumerate(doc_ids, start=1):
                    f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{-rank}\t{run_tag}\n")
        logger.info("Run saved to %s", output_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_k_values(self) -> List[int]:
        k_values = set()
        for m in self.eval_cfg.metrics:
            if "@" in m:
                k_values.add(int(m.split("@")[1]))
        return sorted(k_values) or [10, 100, 1000]

    def _build_result(
        self,
        scores: Dict[str, float],
        model_name: str,
        stage: str,
        kd_mode: str,
    ) -> MetricsResult:
        cfg = self.config
        result = MetricsResult(
            model_name=model_name,
            stage=stage,
            dataset=cfg.data.dataset,
            kd_mode=kd_mode,
            use_lora=cfg.model.use_lora,
            use_hard_negatives=cfg.training.hard_negatives,
            use_matryoshka=cfg.model.use_matryoshka,
        )
        result.mrr_at_10 = scores.get("MRR@10", 0.0)
        result.ndcg_at_10 = scores.get("NDCG@10", 0.0)
        result.map_at_10 = scores.get("MAP@10", 0.0)
        result.recall_at_10 = scores.get("Recall@10", 0.0)
        result.recall_at_100 = scores.get("Recall@100", 0.0)
        result.recall_at_1000 = scores.get("Recall@1000", 0.0)
        result.extra = {k: v for k, v in scores.items() if k not in {
            "MRR@10", "NDCG@10", "MAP@10", "Recall@10", "Recall@100", "Recall@1000"
        }}
        return result
