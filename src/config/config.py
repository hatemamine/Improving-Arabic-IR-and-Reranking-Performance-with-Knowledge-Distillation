from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


SUPPORTED_MODELS = {
    # Bi-encoder base models
    "AraELECTRA": "aubmindlab/araelectra-base-discriminator",
    "AraDPR":     "abdoelsayed/AraDPR",
    "mMiniLML":   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # Cross-encoder models (also usable as KD teacher)
    "mMiniLMv2CE": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
}

SUPPORTED_DATASETS = {
    "mmarco": {
        "hf_id": "hatemestinbejaia/ExperimentDATA_knowledge_distillation_vs_fine_tuning",
        "qrels": "qrels.msmarco-passage.dev-subset.txt",
        "queries": "queries-arabic-preprocess-for-araelectra",
        "collection": "collection-arabic-preprocess-for-araelectra",
        "dev_samples": "dev_samples_araelectra.pkl",
    },
    "mrtydi": {
        "hf_id": "hatemestinbejaia/ExperimentDATA_knowledge_distillation_vs_fine_tuning",
        "qrels": "qrels.test.mrtydi.txt",
        "queries": "queries-arabic-preprocess-for-araelectra",
        "collection": "collection-arabic-preprocess-for-araelectra",
        "dev_samples": "dev_samples_araelectra.pkl",
    },
}


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    # Set to None to auto-detect per model architecture
    target_modules: Optional[List[str]] = None
    lora_dropout: float = 0.1
    bias: str = "none"

    def get_target_modules(self, model_name: str) -> List[str]:
        if self.target_modules is not None:
            return self.target_modules
        # Sensible defaults per architecture
        if "electra" in model_name.lower():
            return ["query", "key", "value"]
        if "bert" in model_name.lower() or "dpr" in model_name.lower():
            return ["query", "key", "value", "dense"]
        return ["q_proj", "k_proj", "v_proj"]


@dataclass
class DataConfig:
    dataset: str = "mmarco"                          # "mmarco" | "mrtydi"
    hf_dataset_id: str = ""                          # auto-filled from SUPPORTED_DATASETS
    train_subset: str = "KD_train_dataset_t05M_t15m_eval_10k_preprocessedAraELECTRA"
    dev_samples_file: str = "dev_samples_araelectra.pkl"
    max_train_samples: Optional[int] = None          # None = use all
    num_dev_queries: int = 300
    num_max_dev_negatives: int = 300
    cache_dir: str = "./data_cache"

    def __post_init__(self):
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"dataset must be one of {list(SUPPORTED_DATASETS)}")
        if not self.hf_dataset_id:
            self.hf_dataset_id = SUPPORTED_DATASETS[self.dataset]["hf_id"]


@dataclass
class ModelConfig:
    base_model: str = "AraDPR"                       # "AraELECTRA" | "AraDPR" | "mMiniLML"
    encoder_type: str = "bi"                         # "bi" | "cross" | "both"
    max_length: int = 256
    use_lora: bool = False
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    # Matryoshka dims (bi-encoder only, ignored if use_matryoshka=False)
    use_matryoshka: bool = False
    matryoshka_dims: List[int] = field(default_factory=lambda: [768, 512, 256, 128])

    def get_hf_model_id(self) -> str:
        if self.base_model not in SUPPORTED_MODELS:
            # Allow passing a raw HuggingFace model ID directly
            return self.base_model
        return SUPPORTED_MODELS[self.base_model]


@dataclass
class TrainingConfig:
    # KD settings
    use_kd: bool = True
    kd_mode: str = "pairwise"        # "pairwise" | "listwise"  (cross-encoder)
    kd_lambda: float = 1.0           # λ in L = λ·L_distillation + (1-λ)·L_student
    #                                  λ=1 → pure KD, λ=0 → vanilla fine-tuning
    kd_temperature: float = 1.0      # softmax temperature for soft labels

    # Hard negatives
    hard_negatives: bool = True
    hn_strategy: str = "curriculum"  # "bm25" | "ance" | "combined" | "curriculum"
    hn_bm25_top_k: int = 200         # top-K BM25 results to mine from
    hn_ance_top_k: int = 200         # top-K ANCE results to mine from
    hn_refresh_steps: int = 5000     # ANCE index refresh frequency (steps)

    # Curriculum schedule: list of (epoch, strategy) tuples
    curriculum_schedule: List[tuple] = field(
        default_factory=lambda: [(1, "bm25"), (2, "combined"), (3, "ance")]
    )

    # Hybrid bi-encoder loss
    use_mnrl_hybrid: bool = False
    mnrl_weight: float = 0.5         # weight of MNRL in (mnrl_weight*MNRL + (1-mnrl_weight)*MarginMSE)

    # Standard training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 8
    learning_rate: float = 7e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.07
    fp16: bool = True
    eval_steps: int = 2000
    save_total_limit: int = 2
    output_dir: str = "./training_output"
    report_to: Optional[str] = None  # None | "wandb" | "tensorboard"
    run_name: str = "arabic-ir-pipeline"

    # W&B — only used when report_to="wandb"
    wandb_project: str = "arabic-ir-kd"
    wandb_entity: Optional[str] = None   # W&B team/username, None = personal account
    wandb_api_key: Optional[str] = None  # set here or via WANDB_API_KEY env var
    wandb_log_artifacts: bool = True      # upload HTML report + model checkpoints as artifacts


@dataclass
class EvalConfig:
    metrics: List[str] = field(
        default_factory=lambda: [
            "mrr@10", "ndcg@10", "recall@10", "recall@100", "recall@1000", "map@10"
        ]
    )
    eval_batch_size: int = 64
    first_stage_top_k: int = 1000    # number of candidates from first-stage retrieval
    rerank_top_k: int = 1000         # number of candidates fed to reranker


@dataclass
class PipelineConfig:
    """Top-level config — fill this in the main notebook config cell."""

    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # Report
    report_output: str = "report.html"
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_repo_id: Optional[str] = None   # e.g. "yourusername/my-arabic-bi-encoder"

    # Reproducibility
    seed: int = 42

    @property
    def use_wandb(self) -> bool:
        return self.training.report_to == "wandb"

    def summary(self) -> str:
        wandb_info = (
            f"wandb: project={self.training.wandb_project}"
            + (f", entity={self.training.wandb_entity}" if self.training.wandb_entity else "")
            if self.use_wandb else "disabled"
        )
        lines = [
            "=== Pipeline Configuration ===",
            f"Dataset      : {self.data.dataset}",
            f"Base model   : {self.model.base_model} ({self.model.get_hf_model_id()})",
            f"Encoder type : {self.model.encoder_type}",
            f"KD           : {self.training.use_kd} (mode={self.training.kd_mode}, λ={self.training.kd_lambda}, T={self.training.kd_temperature})",
            f"Hard negs    : {self.training.hard_negatives} (strategy={self.training.hn_strategy})",
            f"LoRA         : {self.model.use_lora}",
            f"Matryoshka   : {self.model.use_matryoshka} (dims={self.model.matryoshka_dims})",
            f"MNRL hybrid  : {self.training.use_mnrl_hybrid} (weight={self.training.mnrl_weight})",
            f"Epochs       : {self.training.num_train_epochs}",
            f"Batch size   : {self.training.per_device_train_batch_size} (accum={self.training.gradient_accumulation_steps})",
            f"LR           : {self.training.learning_rate}",
            f"Metrics      : {self.evaluation.metrics}",
            f"Report       : {self.report_output}",
            f"W&B          : {wandb_info}",
        ]
        return "\n".join(lines)
