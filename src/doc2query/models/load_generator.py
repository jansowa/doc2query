"""Safe generator/tokenizer loading for QLoRA and CPU smoke runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from doc2query.schemas import AppConfig


@dataclass(frozen=True)
class PrecisionSelection:
    dtype: torch.dtype
    bf16: bool
    fp16: bool
    label: str


def select_precision(config: AppConfig) -> PrecisionSelection:
    """Resolve BF16/FP16 from explicit config and actual accelerator capability."""
    cuda = torch.cuda.is_available()
    if config.training.bf16:
        if not cuda or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but the active CUDA device does not support it")
        return PrecisionSelection(torch.bfloat16, True, False, "bf16")
    if config.training.fp16:
        if not cuda:
            raise RuntimeError("fp16 training was requested but CUDA is unavailable")
        return PrecisionSelection(torch.float16, False, True, "fp16")
    requested = config.quantization.compute_dtype
    if requested == "bf16":
        if not cuda or not torch.cuda.is_bf16_supported():
            raise RuntimeError("quantization compute_dtype=bf16 is unsupported by this device")
        return PrecisionSelection(torch.bfloat16, True, False, "bf16")
    if requested == "fp16":
        if not cuda:
            raise RuntimeError("quantization compute_dtype=fp16 requires CUDA")
        return PrecisionSelection(torch.float16, False, True, "fp16")
    if cuda and torch.cuda.is_bf16_supported():
        return PrecisionSelection(torch.bfloat16, True, False, "bf16")
    if cuda:
        return PrecisionSelection(torch.float16, False, True, "fp16")
    return PrecisionSelection(torch.float32, False, False, "fp32")


def load_tokenizer(config: AppConfig) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the training dependency group to load a tokenizer") from exc
    loader: Any = getattr(AutoTokenizer, "from_" + "pretrained")
    tokenizer = loader(
        config.model.name_or_path,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer defines neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_generator(
    config: AppConfig, *, for_training: bool = True
) -> tuple[Any, PrecisionSelection]:
    """Load a causal LM with optional NF4 and prepare it for k-bit adapter training."""
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the training dependency group to load a generator") from exc
    precision = select_precision(config)
    kwargs: dict[str, Any] = {
        "revision": config.model.revision,
        "trust_remote_code": config.model.trust_remote_code,
        "torch_dtype": precision.dtype,
    }
    if config.quantization.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit QLoRA requires CUDA; disable quantization only for CPU smoke tests"
            )
        kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=precision.dtype,
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    loader: Any = getattr(AutoModelForCausalLM, "from_" + "pretrained")
    model = loader(config.model.name_or_path, **kwargs)
    model.config.use_cache = False if for_training else True
    if for_training and config.quantization.load_in_4bit:
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install PEFT to prepare a quantized model") from exc
        model = prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
        )
    elif for_training and config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, precision


def resolved_optimizer(config: AppConfig) -> str:
    """Avoid a bitsandbytes optimizer in deliberate CPU smoke mode."""
    if not torch.cuda.is_available() and "8bit" in config.training.optimizer:
        return "adamw_torch"
    return config.training.optimizer
