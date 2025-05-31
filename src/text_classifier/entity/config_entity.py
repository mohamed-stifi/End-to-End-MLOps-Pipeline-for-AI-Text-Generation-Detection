from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    status_file: str
    required_files: List[str]

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: str
    max_length: int
    batch_size: int
    test_size: float
    val_size: float

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_name: str
    num_train_epochs: int
    warmup_ratio: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: int
    gradient_accumulation_steps: int
    learning_rate: float

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    comparison_file: Path
    # model_path: Path
    # tokenizer_path: Path
    metric_file_name: Path
    # mlflow_uri: str


@dataclass
class HyperparameterOptimizationConfig:
    root_dir: Path
    data_path: Path                   # Path to data_transformation outputs (train/val splits)
    model_types_to_tune: List[str]    # e.g., ["lstm", "bert"] or ["best_model_from_training"]
    n_trials: int                     # Number of HPO trials per model type
    metric_to_optimize: str           # e.g., "val_acc" or "val_loss"
    direction: str                    # "minimize" or "maximize"
    study_name_prefix: str            # Prefix for Optuna study names
    # Specific HPO params for each model type will be in params.yaml
    hpo_params_ranges: Dict[str, Dict[str, Any]] # Loaded from params.yaml
    mlflow_tracking_uri: str          # To log HPO trials
    trainer_hpo_config: Dict[str, Any]# Common PyTorch Lightning Trainer args for HPO trials
