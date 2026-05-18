from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

from transformers import AutoTokenizer


# Arabic unicode ranges and diacritics
_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]")
_TATWEEL = re.compile(r"ـ")
_ALEF_VARIANTS = re.compile(r"[إأآا]")
_YA_VARIANTS = re.compile(r"[يى]")
_TA_MARBUTA = re.compile(r"ة")
_PUNCTUATION = re.compile(r"[^\w\s؀-ۿ]")


class ArabicPreprocessor:
    """
    Light Arabic text normalisation used before tokenisation.

    Applies: diacritic removal → tatweel removal → alef normalisation →
    ya/ta-marbuta normalisation → punctuation removal → whitespace collapse.
    All steps are individually togglable.
    """

    def __init__(
        self,
        remove_diacritics: bool = True,
        remove_tatweel: bool = True,
        normalize_alef: bool = True,
        normalize_ya: bool = True,
        normalize_ta_marbuta: bool = True,
        remove_punctuation: bool = False,
        lowercase: bool = False,
    ):
        self.remove_diacritics = remove_diacritics
        self.remove_tatweel = remove_tatweel
        self.normalize_alef = normalize_alef
        self.normalize_ya = normalize_ya
        self.normalize_ta_marbuta = normalize_ta_marbuta
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFC", text)
        if self.remove_diacritics:
            text = _ARABIC_DIACRITICS.sub("", text)
        if self.remove_tatweel:
            text = _TATWEEL.sub("", text)
        if self.normalize_alef:
            text = _ALEF_VARIANTS.sub("ا", text)
        if self.normalize_ya:
            text = _YA_VARIANTS.sub("ي", text)
        if self.normalize_ta_marbuta:
            text = _TA_MARBUTA.sub("ه", text)
        if self.remove_punctuation:
            text = _PUNCTUATION.sub(" ", text)
        if self.lowercase:
            text = text.lower()
        text = " ".join(text.split())
        return text

    def normalize_batch(self, texts: List[str]) -> List[str]:
        return [self.normalize(t) for t in texts]


class TokenizerWrapper:
    """Wraps HuggingFace tokenizer with Arabic preprocessing."""

    def __init__(
        self,
        model_name: str,
        preprocessor: Optional[ArabicPreprocessor] = None,
        max_length: int = 256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.preprocessor = preprocessor or ArabicPreprocessor()
        self.max_length = max_length

    def __call__(self, texts: List[str], **kwargs):
        normalized = self.preprocessor.normalize_batch(texts)
        return self.tokenizer(
            normalized,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            **kwargs,
        )

    def encode(self, text: str) -> str:
        """Return preprocessed text string (for use with sentence-transformers)."""
        return self.preprocessor.normalize(text)
