from __future__ import annotations
from typing import List, Tuple, Iterable, Optional
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_FINBERT_MODEL_ID = os.getenv("FINBERT_MODEL_ID", "ProsusAI/finbert")
_HF_CACHE_DIR = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or None


class FinBertScorer:
    """
    간단 감성 점수기 (FinBERT)
      - 반환: [(p_neg, p_neu, p_pos, score), ...]
      - score = p_pos - p_neg  ∈ [-1, 1]
    """
    def __init__(self, device: Optional[str] = None, model_id: Optional[str] = None, cache_dir: Optional[str] = _HF_CACHE_DIR):
        model_id = model_id or _FINBERT_MODEL_ID
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=cache_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score_texts(self, texts: Iterable[str], batch_size: int = 16, max_length: int = 256) -> List[Tuple[float, float, float, float]]:
        """텍스트 리스트 감성 점수 계산"""
        texts = list(texts or [])
        out: List[Tuple[float, float, float, float]] = []
        if not texts:
            return out

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)
            logits = self.model(**enc).logits  # [B, 3] (neg, neu, pos)
            probs = F.softmax(logits, dim=-1)  # [B, 3]

            for p in probs.detach().cpu().tolist():
                p_neg, p_neu, p_pos = (float(p[0]), float(p[1]), float(p[2]))
                score = float(p_pos - p_neg)
                out.append((p_neg, p_neu, p_pos, score))
        return out

    @torch.inference_mode()
    def score_one(self, text: str, **kw) -> Tuple[float, float, float, float]:
        """단일 문장 감정 점수"""
        return self.score_texts([text], **kw)[0]
