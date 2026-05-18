from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sentence_transformers import SentenceTransformer, util


# ---------------------------------------------------------------------------
# Bi-encoder losses
# ---------------------------------------------------------------------------


class MarginMSELoss(nn.Module):
    """
    Pairwise KD loss for bi-encoder.
    MSE between (sim(q,p+) - sim(q,p-)) and teacher score margin.

    Input columns: query, positive, negative, label=(teacher_pos - teacher_neg)
    """

    def __init__(self, model: SentenceTransformer, similarity_fct=util.pairwise_dot_score):
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.mse = nn.MSELoss()

    def forward(self, sentence_features: Iterable[dict], labels: Tensor) -> Tensor:
        embeddings = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
        scores_pos = self.similarity_fct(embeddings[0], embeddings[1])
        scores_neg = self.similarity_fct(embeddings[0], embeddings[2])
        margin = scores_pos - scores_neg
        return self.mse(margin, labels)


class HybridMNRLMarginMSELoss(nn.Module):
    """
    Hybrid loss combining MultipleNegativesRankingLoss and MarginMSELoss.
    loss = mnrl_weight * MNRL + (1 - mnrl_weight) * MarginMSE

    Trains faster and benefits from in-batch negatives while also
    incorporating teacher soft labels.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        mnrl_weight: float = 0.5,
        similarity_fct=util.cos_sim,
    ):
        super().__init__()
        self.model = model
        self.mnrl_weight = mnrl_weight
        self.similarity_fct = similarity_fct
        self.mse = nn.MSELoss()

    def forward(self, sentence_features: Iterable[dict], labels: Tensor) -> Tensor:
        embeddings = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
        q_emb, pos_emb, neg_emb = embeddings[0], embeddings[1], embeddings[2]

        # MarginMSE component
        scores_pos = util.pairwise_dot_score(q_emb, pos_emb)
        scores_neg = util.pairwise_dot_score(q_emb, neg_emb)
        margin_mse = self.mse(scores_pos - scores_neg, labels)

        # MNRL component — in-batch negatives via full similarity matrix
        sim_matrix = self.similarity_fct(q_emb, pos_emb)  # (B, B)
        targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        mnrl_loss = F.cross_entropy(sim_matrix * 20, targets)  # scale=20 is standard

        return self.mnrl_weight * mnrl_loss + (1 - self.mnrl_weight) * margin_mse


class MatryoshkaLoss(nn.Module):
    """
    Matryoshka Representation Learning loss.
    Applies a base loss at each specified embedding dimension, summed with
    optional per-dimension weights.

    Reference: Kusupati et al. 2022 — Matryoshka Representation Learning
    """

    def __init__(
        self,
        model: SentenceTransformer,
        base_loss: nn.Module,
        matryoshka_dims: List[int],
        matryoshka_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.model = model
        self.base_loss = base_loss
        self.matryoshka_dims = sorted(matryoshka_dims, reverse=True)
        if matryoshka_weights is None:
            self.matryoshka_weights = [1.0] * len(self.matryoshka_dims)
        else:
            assert len(matryoshka_weights) == len(matryoshka_dims)
            self.matryoshka_weights = matryoshka_weights

    def forward(self, sentence_features: Iterable[dict], labels: Tensor) -> Tensor:
        # Get full embeddings from the model
        full_embeddings = [self.model(sf)["sentence_embedding"] for sf in sentence_features]

        total_loss = torch.tensor(0.0, device=full_embeddings[0].device)
        for dim, weight in zip(self.matryoshka_dims, self.matryoshka_weights):
            truncated = [e[:, :dim] for e in full_embeddings]

            # Re-package as sentence_features dicts for base_loss
            class _FakeFeatures:
                def __init__(self, emb):
                    self._emb = emb

            # Patch: call base_loss with truncated embeddings directly
            scores_pos = util.pairwise_dot_score(truncated[0], truncated[1])
            scores_neg = util.pairwise_dot_score(truncated[0], truncated[2])
            margin = scores_pos - scores_neg
            dim_loss = F.mse_loss(margin, labels)
            total_loss = total_loss + weight * dim_loss

        return total_loss / sum(self.matryoshka_weights)


# ---------------------------------------------------------------------------
# Cross-encoder losses
# ---------------------------------------------------------------------------


class CombinedCrossEncoderLoss(nn.Module):
    """
    Pairwise cross-encoder KD loss implementing the paper's Equation 1:
        L_combined = λ · L_distillation + (1 − λ) · L_student

    where:
        L_distillation = BCE(student_logits, sigmoid(teacher_logits / T))
        L_student      = BCE(student_logits, hard_binary_labels)

    Args:
        kd_lambda:   λ=1 → pure knowledge distillation, λ=0 → vanilla fine-tuning
        temperature: softmax temperature applied to teacher logits
    """

    def __init__(self, kd_lambda: float = 1.0, temperature: float = 1.0):
        super().__init__()
        self.kd_lambda = kd_lambda
        self.temperature = temperature

    def forward(
        self,
        student_logits: Tensor,   # (B,) raw logits from student CE
        teacher_logits: Tensor,   # (B,) raw logits / scores from teacher
        hard_labels: Tensor,      # (B,) binary relevance labels {0, 1}
    ) -> Tensor:
        l_distillation = torch.tensor(0.0, device=student_logits.device)
        l_student = torch.tensor(0.0, device=student_logits.device)

        if self.kd_lambda > 0.0:
            teacher_probs = torch.sigmoid(teacher_logits / self.temperature)
            l_distillation = F.binary_cross_entropy_with_logits(student_logits, teacher_probs)

        if self.kd_lambda < 1.0:
            l_student = F.binary_cross_entropy_with_logits(student_logits, hard_labels.float())

        return self.kd_lambda * l_distillation + (1.0 - self.kd_lambda) * l_student


class ListwiseKDLoss(nn.Module):
    """
    Listwise cross-encoder KD loss implementing the paper's Equation 1:
        L_combined = λ · L_distillation + (1 − λ) · L_student

    where:
        L_distillation = KL(softmax(student/T) || softmax(teacher/T))  per query group
        L_student      = BCE(student_logits, hard_binary_labels)

    For each query, given K candidate passages, computes KL divergence
    between the temperature-scaled softmax of teacher scores and student scores.

    Input format: student_scores (B,), teacher_scores (B,), group_sizes list
    where B = sum(group_sizes) and each group corresponds to one query's candidates.

    Args:
        kd_lambda:   λ=1 → pure KD, λ=0 → vanilla fine-tuning
        temperature: softmax temperature
    """

    def __init__(self, temperature: float = 1.0, kd_lambda: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.kd_lambda = kd_lambda

    def forward(
        self,
        student_scores: Tensor,     # (B,) flat across all query groups
        teacher_scores: Tensor,     # (B,) flat across all query groups
        group_sizes: List[int],     # list of K_i for each query in the batch
        hard_labels: Optional[Tensor] = None,  # (B,) optional binary labels
    ) -> Tensor:
        T = self.temperature
        l_distillation = torch.tensor(0.0, device=student_scores.device)
        l_student = torch.tensor(0.0, device=student_scores.device)

        if self.kd_lambda > 0.0:
            total_kl = torch.tensor(0.0, device=student_scores.device)
            offset = 0
            for k in group_sizes:
                s = student_scores[offset : offset + k]
                t = teacher_scores[offset : offset + k]
                log_p_student = F.log_softmax(s / T, dim=0)
                p_teacher = F.softmax(t / T, dim=0)
                total_kl = total_kl + F.kl_div(log_p_student, p_teacher, reduction="sum")
                offset += k
            l_distillation = total_kl * (T ** 2) / len(group_sizes)

        if self.kd_lambda < 1.0 and hard_labels is not None:
            l_student = F.binary_cross_entropy_with_logits(student_scores, hard_labels.float())

        return self.kd_lambda * l_distillation + (1.0 - self.kd_lambda) * l_student


class LabelSmoothKDLoss(nn.Module):
    """Deprecated: use CombinedCrossEncoderLoss instead."""

    def __init__(self, alpha: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        hard_labels: Tensor,
    ) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(student_logits, hard_labels.float())
        T = self.temperature
        soft_student = F.log_softmax(student_logits / T, dim=0)
        soft_teacher = F.softmax(teacher_logits / T, dim=0)
        kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
        return self.alpha * bce + (1 - self.alpha) * kl
