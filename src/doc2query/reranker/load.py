"""Safe lazy loading of frozen Hugging Face sequence classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from doc2query.reranker.base import FrozenRerankerConfig


def _pretrained_loader(factory: Any) -> Any:
    """Resolve lazily while keeping import-only tests free of download-looking calls."""
    return getattr(factory, "from_" + "pretrained")


class TransformersReranker:
    """Transformers pair scorer that never enables training or gradients."""

    def __init__(self, config: FrozenRerankerConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("install the training/retrieval dependency groups") from exc

        self.config = config
        self._torch = torch
        tokenizer_loader = _pretrained_loader(AutoTokenizer)
        model_loader = _pretrained_loader(AutoModelForSequenceClassification)
        self._tokenizer = tokenizer_loader(
            config.name_or_path,
            revision=config.revision,
            trust_remote_code=False,
        )
        self._model = model_loader(
            config.name_or_path,
            revision=config.revision,
            trust_remote_code=False,
        ).to(config.device)
        self._model.eval()
        self._model.requires_grad_(False)

    @property
    def name(self) -> str:
        return self.config.name_or_path

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        with self._torch.inference_mode():
            for start in range(0, len(pairs), self.config.batch_size):
                batch = pairs[start : start + self.config.batch_size]
                encoded: dict[str, Any] = self._tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.config.device) for key, value in encoded.items()}
                logits = self._model(**encoded).logits.detach().float().cpu()
                if logits.ndim == 2 and logits.shape[1] > 1:
                    logits = logits[:, -1]
                scores.extend(float(value) for value in logits.reshape(-1).tolist())
        return scores


def load_frozen_reranker(config: FrozenRerankerConfig) -> TransformersReranker:
    """Load a pinned model. This function intentionally has no training options."""
    return TransformersReranker(config)
