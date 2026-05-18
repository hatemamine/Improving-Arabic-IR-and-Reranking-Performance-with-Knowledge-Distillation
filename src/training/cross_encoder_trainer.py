from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
)

from src.config.config import PipelineConfig
from src.models.cross_encoder import CrossEncoderModel
from src.training.losses import LabelSmoothKDLoss, ListwiseKDLoss

logger = logging.getLogger(__name__)


class CrossEncoderTrainer:
    """
    Trains a cross-encoder with one of four modes:
      - NoKD        : standard BCE with binary relevance labels
      - KD pairwise : LabelSmoothKDLoss (α*BCE + (1-α)*KL per pair)
      - KD listwise : ListwiseKDLoss (KL over full K-list per query)
      - Any mode can be combined with LoRA via ModelConfig
    """

    def __init__(self, config: PipelineConfig, cross_encoder: CrossEncoderModel):
        self.config = config
        self.cross_encoder = cross_encoder

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def train(self, train_dataset: Dataset, dev_samples: dict):
        cfg = self.config
        t_cfg = cfg.training

        output_dir = os.path.join(t_cfg.output_dir, "cross_encoder")
        os.makedirs(output_dir, exist_ok=True)

        evaluator = CERerankingEvaluator(dev_samples, name="dev", write_csv=False)

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy=IntervalStrategy.STEPS,
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
            metric_for_best_model="MRR@10",
            greater_is_better=True,
            load_best_model_at_end=True,
            report_to=t_cfg.report_to or "none",
            run_name=t_cfg.run_name + "-cross-encoder",
            seed=cfg.seed,
        )

        compute_metrics_fn = self._make_compute_metrics(output_dir, evaluator)

        trainer = Trainer(
            self.cross_encoder.model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=self._make_eval_dataset(train_dataset),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
            tokenizer=self.cross_encoder.tokenizer,
            compute_metrics=compute_metrics_fn,
        )

        if t_cfg.use_kd:
            trainer = self._patch_loss(trainer, t_cfg)

        logger.info("Starting cross-encoder training (mode=%s)…", t_cfg.kd_mode if t_cfg.use_kd else "nokd")
        trainer.train()
        return trainer

    # ------------------------------------------------------------------
    # Loss patching for KD modes
    # ------------------------------------------------------------------

    def _patch_loss(self, trainer: Trainer, t_cfg) -> Trainer:
        """
        Replace the default HuggingFace compute_loss with a KD loss.
        We monkey-patch compute_loss on the trainer instance.
        """
        if t_cfg.kd_mode == "listwise":
            kd_loss_fn = ListwiseKDLoss(
                temperature=t_cfg.kd_temperature,
                alpha=t_cfg.kd_alpha,
            )
            logger.info("Using ListwiseKDLoss (T=%.2f, α=%.2f)", t_cfg.kd_temperature, t_cfg.kd_alpha)
        else:
            kd_loss_fn = LabelSmoothKDLoss(
                alpha=t_cfg.kd_alpha,
                temperature=t_cfg.kd_temperature,
            )
            logger.info("Using LabelSmoothKDLoss (T=%.2f, α=%.2f)", t_cfg.kd_temperature, t_cfg.kd_alpha)

        kd_loss_fn = kd_loss_fn.to(next(self.cross_encoder.model.parameters()).device)

        original_compute_loss = trainer.compute_loss

        def kd_compute_loss(model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels", None)
            teacher_scores = inputs.pop("teacher_scores", None)

            outputs = model(**inputs)
            student_logits = outputs.logits.squeeze(-1)

            if teacher_scores is not None and t_cfg.kd_mode == "listwise":
                group_sizes = inputs.get("group_sizes", [len(student_logits)])
                hard_labels = labels.float() if labels is not None else None
                loss = kd_loss_fn(student_logits, teacher_scores, group_sizes, hard_labels)
            elif teacher_scores is not None:
                hard_labels = labels.float() if labels is not None else torch.zeros_like(student_logits)
                loss = kd_loss_fn(student_logits, teacher_scores, hard_labels)
            else:
                # Fallback to standard BCE
                loss = F.binary_cross_entropy_with_logits(student_logits, labels.float())

            return (loss, outputs) if return_outputs else loss

        trainer.compute_loss = kd_compute_loss
        return trainer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_compute_metrics(self, output_dir: str, evaluator):
        trainer_ref = self

        def compute_metrics(eval_pred):
            save_path = os.path.join(output_dir, "eval_model")
            trainer_ref.cross_encoder.save(save_path)
            ce_model = CrossEncoder(save_path, max_length=trainer_ref.config.model.max_length)
            mrr10 = evaluator(ce_model)
            return {"MRR@10": mrr10}

        return compute_metrics

    @staticmethod
    def _make_eval_dataset(train_dataset: Dataset) -> Optional[Dataset]:
        if "test" in train_dataset:
            return train_dataset["test"]
        return None
