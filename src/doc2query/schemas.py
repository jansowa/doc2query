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


class TrainingConfig(StrictModel):
    max_length: int = Field(default=1024, ge=64, le=32768)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1.0)
    bf16: bool = False
    fp16: bool = False

    @model_validator(mode="after")
    def precision_is_unambiguous(self) -> "TrainingConfig":
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot both be enabled")
        return self


class GenerationConfig(StrictModel):
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    num_return_sequences: int = Field(default=1, ge=1, le=64)


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
    training: TrainingConfig
    generation: GenerationConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
