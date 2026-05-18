from __future__ import annotations

import logging
import os
from typing import Optional

from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, losses
from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import EarlyStoppingCallback

from src.config.config import PipelineConfig
from src.models.bi_encoder import BiEncoderModel
from src.utils import wandb_logger
from src.training.losses import (
    HybridMNRLMarginMSELoss,
    MarginMSELoss,
    MatryoshkaLoss,
)

logger = logging.getLogger(__name__)


class BiEncoderTrainer:
    """
    Trains a bi-encoder with losses governed by kd_lambda:
      - use_kd=False            : pure MNRL (vanilla fine-tuning)
      - use_kd=True, λ=1        : pure MarginMSE (pure KD distillation)
      - use_kd=True, 0 < λ < 1  : λ·MarginMSE + (1-λ)·MNRL (combined)
      - use_kd=True, λ=0        : pure MNRL (equivalent to use_kd=False)
      - Any of the above + Matryoshka wrapper
    """

    def __init__(self, config: PipelineConfig, bi_encoder: BiEncoderModel):
        self.config = config
        self.bi_encoder = bi_encoder
        self.model = bi_encoder.get_sentence_transformer()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: Dataset,
        dev_samples: dict,
        hard_negative_miner=None,
    ):
        cfg = self.config
        t_cfg = cfg.training
        m_cfg = cfg.model

        evaluator = RerankingEvaluator(dev_samples, name="dev")
        loss = self._build_loss()

        # Rename columns so sentence-transformers finds the right keys
        train_dataset = self._prepare_columns(train_dataset)

        training_args = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(t_cfg.output_dir, "bi_encoder"),
            eval_strategy="steps",
            eval_steps=t_cfg.eval_steps,
            save_steps=t_cfg.eval_steps,
            save_total_limit=t_cfg.save_total_limit,
            fp16=t_cfg.fp16,
            per_device_train_batch_size=t_cfg.per_device_train_batch_size,
            gradient_accumulation_steps=t_cfg.gradient_accumulation_steps,
            logging_steps=t_cfg.eval_steps,
            warmup_ratio=t_cfg.warmup_ratio,
            num_train_epochs=t_cfg.num_train_epochs,
            learning_rate=t_cfg.learning_rate,
            weight_decay=t_cfg.weight_decay,
            metric_for_best_model="eval_mrr@10",
            greater_is_better=True,
            load_best_model_at_end=True,
            report_to=t_cfg.report_to or "none",
            run_name=t_cfg.run_name + "-bi-encoder",
            seed=cfg.seed,
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            loss=loss,
            evaluator=evaluator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        )

        if hard_negative_miner is not None and t_cfg.hard_negatives:
            self._register_ance_callback(trainer, hard_negative_miner, t_cfg)

        if cfg.use_wandb:
            self._register_wandb_callback(trainer)

        logger.info("Starting bi-encoder training…")
        trainer.train()
        return trainer

    # ------------------------------------------------------------------
    # Loss construction
    # ------------------------------------------------------------------

    def _build_loss(self):
        cfg = self.config
        t_cfg = cfg.training
        m_cfg = cfg.model

        lam = t_cfg.kd_lambda if t_cfg.use_kd else 0.0

        if lam == 0.0:
            # λ=0 or use_kd=False → vanilla fine-tuning with MNRL
            base_loss = losses.MultipleNegativesRankingLoss(model=self.model)
            logger.info("Bi-encoder loss: MNRL (λ=0, vanilla fine-tuning)")
        elif lam == 1.0:
            # λ=1 → pure KD distillation with MarginMSE
            base_loss = MarginMSELoss(model=self.model)
            logger.info("Bi-encoder loss: MarginMSE (λ=1, pure KD)")
        else:
            # 0 < λ < 1 → λ·MarginMSE + (1-λ)·MNRL
            # HybridMNRLMarginMSELoss uses mnrl_weight for MNRL; map: mnrl_weight = 1-λ
            base_loss = HybridMNRLMarginMSELoss(
                model=self.model,
                mnrl_weight=1.0 - lam,
            )
            logger.info("Bi-encoder loss: λ·MarginMSE + (1-λ)·MNRL (λ=%.2f)", lam)

        if m_cfg.use_matryoshka:
            logger.info("Wrapping loss with Matryoshka dims: %s", m_cfg.matryoshka_dims)
            return MatryoshkaLoss(
                model=self.model,
                base_loss=base_loss,
                matryoshka_dims=m_cfg.matryoshka_dims,
            )

        return base_loss

    # ------------------------------------------------------------------
    # Column helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_columns(dataset: Dataset) -> Dataset:
        """Ensure dataset has 'anchor', 'positive', 'negative' columns."""
        rename_map = {}
        if "query" in dataset.column_names and "anchor" not in dataset.column_names:
            rename_map["query"] = "anchor"
        if rename_map:
            dataset = dataset.rename_columns(rename_map)
        # Drop KD-specific columns not needed for NoKD losses
        cols_to_drop = [c for c in ["pos_score", "neg_score"] if c in dataset.column_names]
        if cols_to_drop:
            dataset = dataset.remove_columns(cols_to_drop)
        return dataset

    # ------------------------------------------------------------------
    # ANCE callback
    # ------------------------------------------------------------------

    def _register_ance_callback(self, trainer, miner, t_cfg):
        from transformers import TrainerCallback

        model_ref = self.model
        refresh_steps = t_cfg.hn_refresh_steps

        class ANCERefreshCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step > 0 and state.global_step % refresh_steps == 0:
                    logger.info("ANCE refresh at step %d", state.global_step)
                    # queries/corpus must be set on miner before training starts
                    if hasattr(miner, "_queries") and hasattr(miner, "_corpus"):
                        miner.refresh_ance_negatives(
                            miner._queries, miner._corpus, model_ref
                        )

        trainer.add_callback(ANCERefreshCallback())

    def _register_wandb_callback(self, trainer):
        from transformers import TrainerCallback

        class WandbMetricsCallback(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics:
                    wandb_logger.log_metrics(
                        {f"bi_encoder/{k}": v for k, v in metrics.items()},
                        step=state.global_step,
                    )

        trainer.add_callback(WandbMetricsCallback())
