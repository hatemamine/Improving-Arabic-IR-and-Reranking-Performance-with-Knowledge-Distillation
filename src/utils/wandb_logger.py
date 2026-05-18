from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_run = None  # module-level W&B run handle


def init(config) -> Optional[Any]:
    """
    Initialise a W&B run from PipelineConfig.
    Returns the run object, or None if W&B is disabled.
    Idempotent: calling init() a second time returns the existing run.
    """
    global _run
    if not config.use_wandb:
        return None
    if _run is not None:
        return _run

    try:
        import wandb
    except ImportError:
        raise ImportError("wandb is required. Install with: pip install wandb")

    t = config.training
    if t.wandb_api_key:
        os.environ["WANDB_API_KEY"] = t.wandb_api_key

    _run = wandb.init(
        project=t.wandb_project,
        entity=t.wandb_entity,
        name=t.run_name,
        config=_flatten_config(config),
        reinit=True,
    )
    logger.info("W&B run initialised: %s", _run.url)
    return _run


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """Log a dict of scalar metrics to the active W&B run."""
    if _run is None:
        return
    _run.log(metrics, step=step)


def log_summary(metrics: Dict[str, float]):
    """Write final metrics to W&B run summary (shown on the run overview page)."""
    if _run is None:
        return
    for k, v in metrics.items():
        _run.summary[k] = v


def log_artifact(
    path: str,
    artifact_name: str,
    artifact_type: str = "result",
    description: str = "",
):
    """Upload a file or directory as a W&B artifact."""
    if _run is None:
        return
    try:
        import wandb
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type, description=description)
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        _run.log_artifact(artifact)
        logger.info("W&B artifact logged: %s (%s)", artifact_name, path)
    except Exception as e:
        logger.warning("Failed to log W&B artifact: %s", e)


def log_table(name: str, columns: List[str], rows: List[list]):
    """Log a W&B Table (useful for metrics comparison across models)."""
    if _run is None:
        return
    try:
        import wandb
        table = wandb.Table(columns=columns, data=rows)
        _run.log({name: table})
    except Exception as e:
        logger.warning("Failed to log W&B table: %s", e)


def finish():
    """Close the W&B run."""
    global _run
    if _run is not None:
        _run.finish()
        _run = None


def log_results(results: list, config):
    """
    Convenience: log a list of MetricsResult objects as a W&B table
    and write best metrics to summary.
    """
    if _run is None:
        return

    if not results:
        return

    rows = [list(r.as_dict().values()) for r in results]
    columns = list(results[0].as_dict().keys())
    log_table("eval_metrics", columns, rows)

    # Write best-per-metric to summary
    num_cols = ["MRR@10", "NDCG@10", "MAP@10", "R@10", "R@100", "R@1000"]
    for col in num_cols:
        best = max((r.as_dict().get(col, 0.0) for r in results), default=0.0)
        _run.summary[f"best_{col}"] = best


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------

def _flatten_config(config) -> dict:
    """Convert nested PipelineConfig into a flat dict for W&B config panel."""
    flat = {}
    d = config.data
    flat.update({
        "dataset": d.dataset,
        "max_train_samples": d.max_train_samples,
    })
    m = config.model
    flat.update({
        "base_model": m.base_model,
        "encoder_type": m.encoder_type,
        "max_length": m.max_length,
        "use_lora": m.use_lora,
        "lora_r": m.lora_config.r if m.use_lora else None,
        "lora_alpha": m.lora_config.lora_alpha if m.use_lora else None,
        "use_matryoshka": m.use_matryoshka,
        "matryoshka_dims": str(m.matryoshka_dims) if m.use_matryoshka else None,
    })
    t = config.training
    flat.update({
        "use_kd": t.use_kd,
        "kd_mode": t.kd_mode,
        "kd_alpha": t.kd_alpha,
        "kd_temperature": t.kd_temperature,
        "hard_negatives": t.hard_negatives,
        "hn_strategy": t.hn_strategy,
        "use_mnrl_hybrid": t.use_mnrl_hybrid,
        "mnrl_weight": t.mnrl_weight,
        "num_train_epochs": t.num_train_epochs,
        "per_device_train_batch_size": t.per_device_train_batch_size,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "learning_rate": t.learning_rate,
        "weight_decay": t.weight_decay,
        "warmup_ratio": t.warmup_ratio,
        "fp16": t.fp16,
        "seed": config.seed,
    })
    return flat
