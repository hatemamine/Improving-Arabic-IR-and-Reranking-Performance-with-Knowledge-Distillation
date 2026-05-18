from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Metric name → MetricsResult attribute name
_METRIC_ATTR = {
    "MRR@10":      "mrr_at_10",
    "NDCG@10":     "ndcg_at_10",
    "MAP@10":      "map_at_10",
    "Recall@10":   "recall_at_10",
    "Recall@100":  "recall_at_100",
    "Recall@1000": "recall_at_1000",
}

# Metrics computed for GR by default
_GR_METRICS = ["MRR@10", "NDCG@10", "Recall@100", "Recall@1000"]


@dataclass
class MetricsResult:
    """Container for retrieval / reranking metrics on a single dataset."""

    model_name: str
    stage: str          # "first_stage" | "reranking_bm25" | "reranking_faiss"
    dataset: str        # "mmarco" | "mrtydi"
    kd_mode: str        # "nokd" | "pairwise_kd" | "listwise_kd"
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

    def get(self, metric: str) -> float:
        """Return a metric value by its display name (e.g. 'MRR@10')."""
        attr = _METRIC_ATTR.get(metric)
        if attr:
            return getattr(self, attr, 0.0)
        return self.extra.get(metric, 0.0)

    def as_dict(self) -> dict:
        return {
            "model":       self.model_name,
            "stage":       self.stage,
            "dataset":     self.dataset,
            "kd_mode":     self.kd_mode,
            "lora":        self.use_lora,
            "hard_neg":    self.use_hard_negatives,
            "matryoshka":  self.use_matryoshka,
            "MRR@10":      round(self.mrr_at_10, 4),
            "NDCG@10":     round(self.ndcg_at_10, 4),
            "MAP@10":      round(self.map_at_10, 4),
            "R@10":        round(self.recall_at_10, 4),
            "R@100":       round(self.recall_at_100, 4),
            "R@1000":      round(self.recall_at_1000, 4),
            **{k: round(v, 4) for k, v in self.extra.items()},
        }


@dataclass
class GeneralizationResult:
    """
    Generalization Ratio (GR) for a model evaluated on two datasets.

    GR(metric) = zero_shot_metric / in_domain_metric

    Values near or above 1.0 indicate the model generalises robustly
    to unseen datasets. GR > 1 means the model actually performs *better*
    on the out-of-domain set.

    Typical setup:
        in_domain  → mMARCO Arabic   (training domain)
        zero_shot  → Mr.TyDi Arabic  (unseen domain)
    """

    model_name: str
    stage: str
    kd_mode: str
    in_domain: MetricsResult
    zero_shot: MetricsResult
    gr: Dict[str, float] = field(default_factory=dict)  # {"GR_MRR@10": 0.93, ...}

    def compute(self, metrics: Optional[List[str]] = None) -> "GeneralizationResult":
        """Compute GR for each requested metric and store in self.gr."""
        for m in (metrics or _GR_METRICS):
            indomain_val = self.in_domain.get(m)
            zeroshot_val = self.zero_shot.get(m)
            self.gr[f"GR_{m}"] = round(zeroshot_val / indomain_val, 4) if indomain_val > 0 else 0.0
        return self

    def as_dict(self) -> dict:
        base = {
            "model":        self.model_name,
            "stage":        self.stage,
            "kd_mode":      self.kd_mode,
            "in_domain_ds": self.in_domain.dataset,
            "zero_shot_ds": self.zero_shot.dataset,
        }
        # In-domain metrics
        for m in _GR_METRICS:
            base[f"in_domain_{m}"] = round(self.in_domain.get(m), 4)
        # Zero-shot metrics
        for m in _GR_METRICS:
            base[f"zero_shot_{m}"] = round(self.zero_shot.get(m), 4)
        # GR values
        base.update(self.gr)
        return base


def compute_generalization_ratio(
    in_domain: MetricsResult,
    zero_shot: MetricsResult,
    metrics: Optional[List[str]] = None,
) -> GeneralizationResult:
    """
    Compute Generalization Ratio between an in-domain and a zero-shot result.

    Args:
        in_domain:  MetricsResult on the training dataset (e.g. mMARCO)
        zero_shot:  MetricsResult on the held-out dataset (e.g. Mr.TyDi)
        metrics:    Which metrics to compute GR for (default: MRR@10, NDCG@10, R@100, R@1000)

    Returns:
        GeneralizationResult with .gr dict populated.
    """
    gr_result = GeneralizationResult(
        model_name=in_domain.model_name,
        stage=in_domain.stage,
        kd_mode=in_domain.kd_mode,
        in_domain=in_domain,
        zero_shot=zero_shot,
    )
    return gr_result.compute(metrics)


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

    mrr:    Dict[int, float] = {k: 0.0 for k in all_k}
    ndcg:   Dict[int, float] = {k: 0.0 for k in all_k}
    ap:     Dict[int, float] = {k: 0.0 for k in all_k}
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
        results[f"MRR@{k}"]    = mrr[k]    / num_queries
        results[f"NDCG@{k}"]   = ndcg[k]   / num_queries
        results[f"MAP@{k}"]    = ap[k]      / num_queries
        results[f"Recall@{k}"] = recall[k]  / num_queries

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
