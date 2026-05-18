from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricsResult:
    """Container for all retrieval/reranking metrics."""

    model_name: str
    stage: str                         # "first_stage" | "reranking"
    dataset: str
    kd_mode: str                       # "nokd" | "pairwise_kd" | "listwise_kd"
    use_lora: bool = False
    use_hard_negatives: bool = False
    use_matryoshka: bool = False

    mrr_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    map_at_10: float = 0.0
    recall_at_10: float = 0.0
    recall_at_100: float = 0.0
    recall_at_1000: float = 0.0

    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "model": self.model_name,
            "stage": self.stage,
            "dataset": self.dataset,
            "kd_mode": self.kd_mode,
            "lora": self.use_lora,
            "hard_neg": self.use_hard_negatives,
            "matryoshka": self.use_matryoshka,
            "MRR@10": round(self.mrr_at_10, 4),
            "NDCG@10": round(self.ndcg_at_10, 4),
            "MAP@10": round(self.map_at_10, 4),
            "R@10": round(self.recall_at_10, 4),
            "R@100": round(self.recall_at_100, 4),
            "R@1000": round(self.recall_at_1000, 4),
            **{k: round(v, 4) for k, v in self.extra.items()},
        }


def compute_metrics(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute MRR@K, NDCG@K, MAP@K, Recall@K from a run dict and qrels.

    Args:
        run:      {qid: [doc_id ranked highest to lowest]}
        qrels:    {qid: {doc_id: relevance_score}}
        k_values: list of cutoffs, default [10, 100, 1000]

    Returns:
        Dict of metric_name → score
    """
    if k_values is None:
        k_values = [10, 100, 1000]

    all_k = sorted(set(k_values))
    max_k = max(all_k)

    mrr: Dict[int, float] = {k: 0.0 for k in all_k}
    ndcg: Dict[int, float] = {k: 0.0 for k in all_k}
    ap: Dict[int, float] = {k: 0.0 for k in all_k}
    recall: Dict[int, float] = {k: 0.0 for k in all_k}

    num_queries = 0
    for qid, ranked_docs in run.items():
        relevant = {str(d): r for d, r in qrels.get(str(qid), {}).items() if r > 0}
        if not relevant:
            continue
        num_queries += 1
        ranked_docs = [str(d) for d in ranked_docs[:max_k]]

        for k in all_k:
            top_k = ranked_docs[:k]
            num_rel = len(relevant)

            # MRR@K
            for rank, doc in enumerate(top_k, start=1):
                if doc in relevant:
                    mrr[k] += 1.0 / rank
                    break

            # NDCG@K
            dcg = sum(
                (2 ** relevant.get(doc, 0) - 1) / math.log2(rank + 1)
                for rank, doc in enumerate(top_k, start=1)
            )
            ideal_rels = sorted(relevant.values(), reverse=True)[:k]
            idcg = sum(
                (2 ** r - 1) / math.log2(rank + 1)
                for rank, r in enumerate(ideal_rels, start=1)
            )
            ndcg[k] += dcg / idcg if idcg > 0 else 0.0

            # MAP@K
            hits, prec_sum = 0, 0.0
            for rank, doc in enumerate(top_k, start=1):
                if doc in relevant:
                    hits += 1
                    prec_sum += hits / rank
            ap[k] += prec_sum / min(num_rel, k) if num_rel > 0 else 0.0

            # Recall@K
            hits_recall = sum(1 for doc in top_k if doc in relevant)
            recall[k] += hits_recall / num_rel if num_rel > 0 else 0.0

    if num_queries == 0:
        return {}

    results = {}
    for k in all_k:
        results[f"MRR@{k}"] = mrr[k] / num_queries
        results[f"NDCG@{k}"] = ndcg[k] / num_queries
        results[f"MAP@{k}"] = ap[k] / num_queries
        results[f"Recall@{k}"] = recall[k] / num_queries

    return results


def parse_trec_eval_output(stdout: str) -> Dict[str, float]:
    """Parse pyserini trec_eval stdout into a metric dict."""
    results = {}
    for line in stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) == 3:
            metric_name, _, value = parts
            try:
                results[metric_name] = float(value)
            except ValueError:
                pass
    return results
