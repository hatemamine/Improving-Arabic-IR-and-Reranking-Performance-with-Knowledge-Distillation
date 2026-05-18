from __future__ import annotations

import logging
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

from src.config.config import ModelConfig

logger = logging.getLogger(__name__)


class BiEncoderModel:
    """
    Wraps SentenceTransformer with optional LoRA (via PEFT) and exposes a
    uniform interface used by the trainer and retriever.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_id = config.get_hf_model_id()
        self.model: SentenceTransformer = self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> SentenceTransformer:
        logger.info("Loading bi-encoder: %s", self.model_id)
        model = SentenceTransformer(self.model_id)

        if self.config.use_lora:
            model = self._apply_lora(model)

        return model

    def _apply_lora(self, model: SentenceTransformer) -> SentenceTransformer:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "peft is required for LoRA. Install with: pip install peft"
            )

        lora_cfg = self.config.lora_config
        target_modules = lora_cfg.get_target_modules(self.model_id)
        logger.info("Applying LoRA (r=%d, α=%d) to modules: %s", lora_cfg.r, lora_cfg.lora_alpha, target_modules)

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
        )

        # Apply to the underlying transformer inside SentenceTransformer
        transformer_module = model[0]
        transformer_module.auto_model = get_peft_model(
            transformer_module.auto_model, peft_config
        )
        transformer_module.auto_model.print_trainable_parameters()
        return model

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def encode(self, sentences, **kwargs):
        return self.model.encode(sentences, **kwargs)

    def save(self, path: str):
        logger.info("Saving bi-encoder to %s", path)
        self.model.save(path)

    def push_to_hub(self, repo_id: str, token: Optional[str] = None):
        self.model.push_to_hub(repo_id, token=token)

    def get_sentence_transformer(self) -> SentenceTransformer:
        return self.model

    @classmethod
    def from_pretrained(cls, path_or_id: str, config: Optional[ModelConfig] = None) -> "BiEncoderModel":
        """Load a saved or hub model without re-applying LoRA."""
        if config is None:
            config = ModelConfig(base_model=path_or_id, use_lora=False)
        instance = object.__new__(cls)
        instance.config = config
        instance.model_id = path_or_id
        instance.model = SentenceTransformer(path_or_id)
        return instance
