from __future__ import annotations

import logging
import operator
import os
from typing import Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer, util

from src.config.config import EvalConfig, PipelineConfig
from src.evaluation.metrics import (
    GeneralizationResult,
    MetricsResult,
    compute_generalization_ratio,
    compute_metrics,
)

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """
    End-to-end evaluator for first-stage retrieval and reranking.

    Supports three evaluation scenarios:
      1. evaluate_bi_encoder_top1000()  — bi-encoder FAISS retrieval at K=1000
      2. evaluate_cross_encoder_on_bm25() — cross-encoder reranking of BM25 candidates
      3. evaluate_reranking()           — cross-encoder reranking of any first-stage run

    Generalization Ratio (GR) is computed via compute_generalization_ratio():
      GR(metric) = zero_shot_metric / in_domain_metric
    Values ≥ 1 indicate robust out-of-domain generalisation.

    Typical workflow:
        # In-domain (mMARCO)
        indomain_eval = PipelineEvaluator(cfg, mmarco_queries, corpus, mmarco_qrels)
        fs_mmarco = indomain_eval.evaluate_bi_encoder_top1000(bi_enc, "AraDPR", "kd")

        # Zero-shot (Mr.TyDi)
        zeroshot_eval = PipelineEvaluator(cfg, mrtydi_queries, corpus, mrtydi_qrels)
        fs_mrtydi = zeroshot_eval.evaluate_bi_encoder_top1000(bi_enc, "AraDPR", "kd")

        gr = compute_generalization_ratio(fs_mmarco, fs_mrtydi)
    """

    def __init__(
        self,
        config: PipelineConfig,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        dataset_name: Optional[str] = None,
    ):
        self.config = config
        self.queries = queries
        self.corpus = corpus
        self.qrels = qrels
        self.dataset_name = dataset_name or config.data.dataset
        self.eval_cfg: EvalConfig = config.evaluation

    # ------------------------------------------------------------------
    # Scenario 1 — Bi-encoder first-stage retrieval at K=1000
    # ------------------------------------------------------------------

    def evaluate_bi_encoder_top1000(
        self,
        bi_encoder,
        model_name: str = "bi_encoder",
        kd_mode: str = "nokd",
        top_k: int = 1000,
        batch_size: int = 128,
    ) -> MetricsResult:
        """
        Encode all queries and retrieve top-K from a FAISS index built on the corpus.
        Returns a MetricsResult with stage='first_stage'.

        Args:
            bi_encoder: SentenceTransformer (or BiEncoderModel) instance
            model_name: display label
            kd_mode:    'nokd' | 'pairwise_kd' | 'listwise_kd'
            top_k:      number of candidates to retrieve (default 1000)
            batch_size: encoding batch size
        """
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError("faiss is required. Install with: pip install faiss-gpu (or faiss-cpu)")

        encoder = bi_encoder.model if hasattr(bi_encoder, "model") else bi_encoder

        logger.info("[bi-encoder @%d] encoding corpus (%d docs)…", top_k, len(self.corpus))
        doc_ids = list(self.corpus.keys())
        doc_texts = [self.corpus[d] for d in doc_ids]
        corpus_embs = encoder.encode(
            doc_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        dim = corpus_embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_embs)

        logger.info("[bi-encoder @%d] encoding %d queries…", top_k, len(self.queries))
        qids = list(self.queries.keys())
        q_texts = [self.queries[q] for q in qids]
        q_embs = encoder.encode(
            q_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        _, indices = index.search(q_embs, top_k)

        run: Dict[str, List[str]] = {}
        for qid, idxs in zip(qids, indices):
            run[str(qid)] = [str(doc_ids[i]) for i in idxs if i >= 0]

        k_values = self._parse_k_values()
        scores = compute_metrics(run, self.qrels, k_values)
        result = self._build_result(scores, model_name, "first_stage", kd_mode)
        self._log_result(result, f"bi-encoder @{top_k}")
        return result

    # ------------------------------------------------------------------
    # Scenario 2 — Cross-encoder reranking of BM25 candidates
    # ------------------------------------------------------------------

    def evaluate_cross_encoder_on_bm25(
        self,
        cross_encoder,
        bm25_run: Dict[str, List[str]],
        model_name: str = "cross_encoder",
        kd_mode: str = "nokd",
        top_k: Optional[int] = None,
    ) -> MetricsResult:
        """
        Rerank a BM25 candidate list with a cross-encoder and evaluate.
        Returns a MetricsResult with stage='reranking_bm25'.

        Args:
            cross_encoder: CrossEncoder or CrossEncoderModel instance
            bm25_run:      {qid: [doc_id, ...]} from BM25 retrieval
            model_name:    display label
            kd_mode:       'nokd' | 'pairwise_kd' | 'listwise_kd'
            top_k:         how many BM25 candidates to rerank (default: eval_cfg.rerank_top_k)
        """
        k = top_k or self.eval_cfg.rerank_top_k
        logger.info("[CE on BM25 @%d] reranking %d queries…", k, len(bm25_run))
        reranked = self._rerank(cross_encoder, bm25_run, k, use_cross_encoder=True)
        k_values = self._parse_k_values()
        scores = compute_metrics(reranked, self.qrels, k_values)
        result = self._build_result(scores, model_name, "reranking_bm25", kd_mode)
        self._log_result(result, f"CE on BM25 @{k}")
        return result

    # ------------------------------------------------------------------
    # Scenario 3 — Generic first-stage and reranking evaluation
    # ------------------------------------------------------------------

    def evaluate_first_stage(
        self,
        run: Dict[str, List[str]],
        model_name: str = "bi_encoder",
        kd_mode: str = "nokd",
    ) -> MetricsResult:
        """Evaluate any pre-built retrieval run."""
        k_values = self._parse_k_values()
        scores = compute_metrics(run, self.qrels, k_values)
        result = self._build_result(scores, model_name, "first_stage", kd_mode)
        self._log_result(result, "first-stage")
        return result

    def evaluate_reranking(
        self,
        reranker,
        first_stage_run: Dict[str, List[str]],
        model_name: str = "cross_encoder",
        kd_mode: str = "nokd",
        use_cross_encoder: bool = True,
        top_k: Optional[int] = None,
    ) -> MetricsResult:
        """Evaluate cross-encoder reranking of any first-stage run."""
        k = top_k or self.eval_cfg.rerank_top_k
        reranked = self._rerank(reranker, first_stage_run, k, use_cross_encoder)
        k_values = self._parse_k_values()
        scores = compute_metrics(reranked, self.qrels, k_values)
        stage = "reranking_faiss" if use_cross_encoder else "reranking_bi"
        result = self._build_result(scores, model_name, stage, kd_mode)
        self._log_result(result, stage)
        return result

    # ------------------------------------------------------------------
    # Generalization Ratio
    # ------------------------------------------------------------------

    @staticmethod
    def compute_generalization_ratio(
        in_domain: MetricsResult,
        zero_shot: MetricsResult,
        metrics: Optional[List[str]] = None,
    ) -> GeneralizationResult:
        """
        Compute GR = zero_shot_metric / in_domain_metric.

        Args:
            in_domain:  MetricsResult on training dataset (mMARCO)
            zero_shot:  MetricsResult on held-out dataset (Mr.TyDi)
            metrics:    which metrics to compute GR for (default: MRR@10, NDCG@10, R@100, R@1000)

        Returns:
            GeneralizationResult with .gr dict, e.g. {"GR_MRR@10": 0.92, ...}
        """
        gr = compute_generalization_ratio(in_domain, zero_shot, metrics)
        logger.info(
            "[GR] %s | %s → %s",
            in_domain.model_name,
            {k: f"{v:.3f}" for k, v in gr.gr.items()},
            "robust" if all(v >= 0.9 for v in gr.gr.values()) else "check generalisation",
        )
        return gr

    # ------------------------------------------------------------------
    # Reranking internals
    # ------------------------------------------------------------------

    def _rerank(
        self,
        reranker,
        first_stage_run: Dict[str, List[str]],
        top_k: int,
        use_cross_encoder: bool,
    ) -> Dict[str, List[str]]:
        reranked: Dict[str, List[str]] = {}
        for qid, doc_ids in first_stage_run.items():
            query_text = self.queries.get(str(qid), "")
            candidates = doc_ids[:top_k]

            if use_cross_encoder:
                pairs = [[query_text, self.corpus.get(str(d), "")] for d in candidates]
                with torch.no_grad():
                    scores = reranker.predict(pairs, batch_size=self.eval_cfg.eval_batch_size)
            else:
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

    def save_run(self, run: Dict[str, List[str]], output_path: str, run_tag: str = "run"):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for qid, doc_ids in run.items():
                for rank, doc_id in enumerate(doc_ids, start=1):
                    f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{-rank}\t{run_tag}\n")
        logger.info("Run saved to %s", output_path)

    # ------------------------------------------------------------------
    # Internals
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
            dataset=self.dataset_name,
            kd_mode=kd_mode,
            use_lora=cfg.model.use_lora,
            use_hard_negatives=cfg.training.hard_negatives,
            use_matryoshka=cfg.model.use_matryoshka,
        )
        result.mrr_at_10    = scores.get("MRR@10",      0.0)
        result.ndcg_at_10   = scores.get("NDCG@10",     0.0)
        result.map_at_10    = scores.get("MAP@10",       0.0)
        result.recall_at_10  = scores.get("Recall@10",  0.0)
        result.recall_at_100 = scores.get("Recall@100", 0.0)
        result.recall_at_1000 = scores.get("Recall@1000", 0.0)
        result.extra = {k: v for k, v in scores.items() if k not in {
            "MRR@10", "NDCG@10", "MAP@10", "Recall@10", "Recall@100", "Recall@1000"
        }}
        return result

    @staticmethod
    def _log_result(result: MetricsResult, tag: str):
        logger.info(
            "[%s] %s | MRR@10=%.4f  NDCG@10=%.4f  R@100=%.4f  R@1000=%.4f",
            tag, result.model_name,
            result.mrr_at_10, result.ndcg_at_10,
            result.recall_at_100, result.recall_at_1000,
        )
