from __future__ import annotations

import logging
from typing import List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.config.config import ModelConfig

logger = logging.getLogger(__name__)


class CrossEncoderModel:
    """
    Wraps a HuggingFace sequence-classification model for cross-encoder
    reranking, with optional LoRA support.

    Exposed interface mirrors sentence-transformers CrossEncoder for
    drop-in compatibility with evaluators.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_id = config.get_hf_model_id()
        self.max_length = config.max_length
        self.tokenizer, self.model = self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        logger.info("Loading cross-encoder: %s", self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        hf_config = AutoConfig.from_pretrained(self.model_id)
        hf_config.num_labels = 1
        hf_config.problem_type = "multi_label_classification"
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, config=hf_config
        )
        if self.config.use_lora:
            model = self._apply_lora(model)
        return tokenizer, model

    def _apply_lora(self, model):
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("peft is required for LoRA. Install with: pip install peft")

        lora_cfg = self.config.lora_config
        target_modules = lora_cfg.get_target_modules(self.model_id)
        logger.info("Applying LoRA (r=%d, α=%d) to modules: %s", lora_cfg.r, lora_cfg.lora_alpha, target_modules)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    # ------------------------------------------------------------------
    # Inference (mirrors sentence-transformers CrossEncoder interface)
    # ------------------------------------------------------------------

    def predict(
        self,
        sentence_pairs: List[List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> List[float]:
        device = next(self.model.parameters()).device
        self.model.eval()
        scores = []

        iterator = range(0, len(sentence_pairs), batch_size)
        if show_progress_bar:
            import tqdm
            iterator = tqdm.tqdm(iterator, desc="Predicting")

        with torch.no_grad():
            for start in iterator:
                batch = sentence_pairs[start : start + batch_size]
                queries = [p[0] for p in batch]
                passages = [p[1] for p in batch]
                enc = self.tokenizer(
                    queries,
                    passages,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = self.model(**enc).logits.squeeze(-1)
                scores.extend(logits.cpu().tolist())
        return scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        logger.info("Saving cross-encoder to %s", path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def push_to_hub(self, repo_id: str, token: Optional[str] = None):
        self.model.push_to_hub(repo_id, token=token)
        self.tokenizer.push_to_hub(repo_id, token=token)

    @classmethod
    def from_pretrained(cls, path_or_id: str, config: Optional[ModelConfig] = None) -> "CrossEncoderModel":
        if config is None:
            config = ModelConfig(base_model=path_or_id, use_lora=False)
        from sentence_transformers.cross_encoder import CrossEncoder as STCrossEncoder
        # Use sentence-transformers CrossEncoder for loading to stay compatible with evaluators
        instance = object.__new__(cls)
        instance.config = config
        instance.model_id = path_or_id
        instance.max_length = config.max_length
        instance._st_model = STCrossEncoder(path_or_id, max_length=config.max_length)
        # Expose predict via ST model
        instance.predict = instance._st_model.predict
        return instance
