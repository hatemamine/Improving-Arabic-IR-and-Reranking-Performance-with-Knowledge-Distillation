from __future__ import annotations

import gc
import os
import pickle
import ctypes
from typing import Dict, Optional, Tuple

import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download

from src.config.config import DataConfig, SUPPORTED_DATASETS


class DatasetLoader:
    """Loads and caches queries, corpus, train/dev splits for mMARCO or Mr.TyDi."""

    def __init__(self, config: DataConfig):
        self.config = config
        self._dataset_info = SUPPORTED_DATASETS[config.dataset]
        os.makedirs(config.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_queries(self, split: str = "dev") -> Dict[int, str]:
        """Return {query_id: query_text}."""
        ds = load_dataset(
            self.config.hf_dataset_id,
            self._dataset_info["queries"],
            trust_remote_code=True,
        )
        queries: Dict[int, str] = {}
        for row in tqdm.tqdm(ds[split], desc="Loading queries"):
            queries[row["id"]] = row["text"]
        return queries

    def load_corpus(self, split: str = "collection") -> Dict[int, str]:
        """Return {doc_id: doc_text}."""
        ds = load_dataset(
            self.config.hf_dataset_id,
            self._dataset_info["collection"],
            trust_remote_code=True,
        )
        corpus: Dict[int, str] = {}
        for row in tqdm.tqdm(ds[split], desc="Loading corpus"):
            corpus[row["id"]] = row["text"]
        self._free_memory()
        return corpus

    def load_train_dataset(self, subset_name: Optional[str] = None) -> Dataset:
        """Load training dataset (with KD scores or plain triplets)."""
        name = subset_name or self.config.train_subset
        ds = load_dataset(
            self.config.hf_dataset_id,
            name,
            trust_remote_code=True,
        )
        train_split = ds["train0"] if "train0" in ds else ds["train"]
        if self.config.max_train_samples:
            train_split = train_split.select(range(self.config.max_train_samples))
        return train_split

    def load_dev_samples(self) -> dict:
        """Load pickled dev_samples dict used by sentence-transformers evaluators."""
        local_path = os.path.join(self.config.cache_dir, self.config.dev_samples_file)
        if not os.path.exists(local_path):
            hf_hub_download(
                repo_id=self.config.hf_dataset_id,
                filename=self.config.dev_samples_file,
                local_dir=self.config.cache_dir,
                repo_type="dataset",
            )
        with open(local_path, "rb") as f:
            return pickle.load(f)

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        """Load relevance judgements. Returns {qid: {doc_id: relevance}}."""
        fname = self._dataset_info["qrels"]
        local_path = os.path.join(self.config.cache_dir, fname)
        if not os.path.exists(local_path):
            hf_hub_download(
                repo_id=self.config.hf_dataset_id,
                filename=fname,
                local_dir=self.config.cache_dir,
                repo_type="dataset",
            )
        qrels: Dict[str, Dict[str, int]] = {}
        with open(local_path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 4:
                    qid, _, doc_id, rel = parts
                else:
                    qid, doc_id, rel = parts[0], parts[2], parts[3]
                qrels.setdefault(qid, {})[doc_id] = int(rel)
        return qrels

    def build_dev_samples_from_triplets(
        self,
        queries: Dict[int, str],
        docs: Dict[int, str],
        triplet_df,
        num_dev_queries: Optional[int] = None,
        num_max_negatives: Optional[int] = None,
    ) -> dict:
        """Build dev_samples dict from a triplets DataFrame (query, pos, neg)."""
        n_q = num_dev_queries or self.config.num_dev_queries
        n_neg = num_max_negatives or self.config.num_max_dev_negatives
        df_shuffle = triplet_df.sample(frac=1, random_state=7).reset_index(drop=True)

        dev_samples: dict = {}
        for _, row in df_shuffle.iterrows():
            qid, pos_id, neg_id = row["query"], row["pos"], row["neg"]
            if qid not in dev_samples and len(dev_samples) < n_q:
                dev_samples[qid] = {
                    "query": queries[qid],
                    "positive": set(),
                    "negative": set(),
                }
            if qid in dev_samples:
                dev_samples[qid]["positive"].add(docs[pos_id])
                if len(dev_samples[qid]["negative"]) < n_neg:
                    dev_samples[qid]["negative"].add(docs[neg_id])
        return dev_samples

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _free_memory():
        gc.collect()
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass
