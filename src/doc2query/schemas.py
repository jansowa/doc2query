"""Validated configuration contracts shared by public commands."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model rejecting misspelled or unsupported fields."""

    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictModel):
    experiment_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    output_dir: Path


class HuggingFaceDatasetSource(StrictModel):
    repo_id: str = Field(min_length=1)
    revision: str = Field(min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$")
    config_name: str = Field(min_length=1)
    split: str = Field(min_length=1)
    private: bool = False
    license_status: Literal["declared", "missing_requires_review"] = "declared"


class DatasetColumnMapping(StrictModel):
    example_id: str = "query_id"
    query: str = "query"
    positive_texts: str = "pos"
    positive_ids: str = "pos_id"
    positive_scores: str = "pos_scores"
    positive_is_synthetic: str = "pos_is_synthetic"
    negative_texts: str = "neg"
    negative_ids: str = "neg_id"
    negative_scores: str = "neg_scores"


class DataConfig(StrictModel):
    input_path: Path | None = None
    input_format: Literal["jsonl", "parquet"] | None = None
    source: HuggingFaceDatasetSource | None = None
    columns: DatasetColumnMapping = Field(default_factory=DatasetColumnMapping)
    eval_path: Path | None = None
    train_split: str = "train"
    eval_split: str = "dev"
    fingerprint: str | None = None
    max_train_examples: int | None = Field(default=None, ge=1)
    max_eval_examples: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def exactly_one_source(self) -> "DataConfig":
        local = self.input_path is not None and self.input_format is not None
        remote = self.source is not None
        if local == remote:
            raise ValueError("configure exactly one of local input_path/input_format or source")
        if (self.input_path is None) != (self.input_format is None):
            raise ValueError("input_path and input_format must be configured together")
        return self


class ModelConfig(StrictModel):
    name_or_path: str = Field(min_length=1)
    revision: str = Field(default="main", min_length=1)
    trust_remote_code: bool = False


class QuantizationConfig(StrictModel):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    compute_dtype: Literal["auto", "bf16", "fp16"] = "auto"


class LoraConfig(StrictModel):
    r: int = Field(default=16, ge=1, le=512)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] | Literal["auto"] = "auto"
    minimum_target_modules: int = Field(default=4, ge=1)
    expected_layer_patterns: list[str] = Field(default_factory=lambda: ["attn", "mlp|proj|fc"])


class TrainingConfig(StrictModel):
    max_length: int = Field(default=1024, ge=64, le=32768)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1.0)
    bf16: bool = False
    fp16: bool = False
    baseline: Literal["b0", "b1"] = "b1"
    strategy: Literal["ordinary", "balanced", "weighted"] = "ordinary"
    num_train_epochs: float = Field(default=1.0, gt=0.0)
    max_steps: int = Field(default=-1, ge=-1)
    gradient_checkpointing: bool = True
    packing: bool = False
    pad_to_max_length: bool = False
    max_completion_tokens: int = Field(default=96, ge=2, le=512)
    min_prompt_tokens: int = Field(default=32, ge=1)
    optimizer: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    logging_steps: int = Field(default=10, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    save_steps: int = Field(default=100, ge=1)
    save_total_limit: int = Field(default=2, ge=1)
    dataloader_num_workers: int = Field(default=0, ge=0)
    weight_min: float = Field(default=0.25, gt=0.0)
    weight_max: float = Field(default=4.0, gt=0.0)
    early_stopping_metric: Literal["eval_loss"] | None = None
    early_stopping_patience: int = Field(default=3, ge=1)
    resume_if_available: bool = False

    @model_validator(mode="after")
    def precision_is_unambiguous(self) -> "TrainingConfig":
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot both be enabled")
        if self.min_prompt_tokens >= self.max_length:
            raise ValueError("min_prompt_tokens must be smaller than max_length")
        if self.weight_min > 1.0 or self.weight_max < 1.0:
            raise ValueError("weight bounds must contain the normalized mean 1.0")
        if self.weight_min > self.weight_max:
            raise ValueError("weight_min must not exceed weight_max")
        if self.packing:
            raise ValueError("packing is not supported with per-example completion-only loss")
        return self


class GenerationConfig(StrictModel):
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    num_return_sequences: int = Field(default=1, ge=1, le=64)
    do_sample: bool = False
    temperature: float = Field(default=0.8, gt=0.0, le=5.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def greedy_has_single_output(self) -> "GenerationConfig":
        if not self.do_sample and self.num_return_sequences != 1:
            raise ValueError("greedy generation supports exactly one return sequence")
        return self


class TrackingConfig(StrictModel):
    backend: Literal["offline", "wandb"] = "offline"
    online: bool = False

    @model_validator(mode="after")
    def backend_matches_mode(self) -> "TrackingConfig":
        if self.online and self.backend == "offline":
            raise ValueError("online tracking requires a non-offline backend")
        return self


class AppConfig(StrictModel):
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    training: TrainingConfig
    generation: GenerationConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
