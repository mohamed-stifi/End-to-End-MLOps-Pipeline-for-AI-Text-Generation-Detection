
from text_classifier.constants import *
from text_classifier.utils.common import read_yaml, create_directories
from text_classifier.entity.config_entity import *
from pathlib import Path
class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_DIR/"config.yaml", params_filepath=CONFIG_DIR/"params.yaml"):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([Path(self.config.artifacts_root)])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            max_sample_number= config.max_sample_number
        )
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            status_file=config.STATUS_FILE,
            required_files=config.required_files
        )
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=self.params.TOKENIZER_NAME,
            max_length=self.params.MAX_LENGTH,
            batch_size=self.params.BATCH_SIZE,
            test_size=self.params.TEST_SIZE,
            val_size=self.params.VAL_SIZE
        )
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        create_directories([config.root_dir])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_name=config.model_name,
            num_train_epochs=params.num_train_epochs,
            warmup_ratio=params.warmup_ratio,
            per_device_train_batch_size=params.per_device_train_batch_size,
            per_device_eval_batch_size=params.per_device_eval_batch_size,
            weight_decay=params.weight_decay,
            logging_steps=params.logging_steps,
            evaluation_strategy=params.evaluation_strategy,
            eval_steps=params.eval_steps,
            save_steps=params.save_steps,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
            learning_rate=params.learning_rate
        )
        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            comparison_file = config.comparison_file,
            #model_path=config.model_path,
            #tokenizer_path=config.tokenizer_path,
            metric_file_name=config.metric_file_name,
            #mlflow_uri="https://dagshub.com/username/mlops-text-classification.mlflow"
        )
        return model_evaluation_config
    
    def get_hyperparameter_optimization_config(self) -> HyperparameterOptimizationConfig:
        config_hpo = self.config.hyperparameter_optimization
        params_hpo = self.params.get("HyperparameterOptimization", {}) # HPO specific params
        create_directories([Path(config_hpo.root_dir)])

        hpo_config = HyperparameterOptimizationConfig(
            root_dir=Path(config_hpo.root_dir),
            data_path=Path(config_hpo.data_path), # From main config
            model_types_to_tune=config_hpo.get("model_types_to_tune", MODEL_TYPES), # Default to all
            n_trials=params_hpo.get("n_trials", 20),
            metric_to_optimize=params_hpo.get("metric_to_optimize", "val_acc"),
            direction=params_hpo.get("direction", "maximize"),
            study_name_prefix=params_hpo.get("study_name_prefix", "text_classifier_hpo"),
            hpo_params_ranges=params_hpo.get("hpo_params_ranges", {}), # Will hold Optuna suggest ranges
            mlflow_tracking_uri=self.config.mlflow_tracking_uri, # Global MLflow URI
            trainer_hpo_config=params_hpo.get("trainer_hpo_config", { # Default trainer args for HPO
                "max_epochs": 3, # Fewer epochs for HPO trials
                "log_every_n_steps": 50,
                "enable_checkpointing": False, # Usually disable checkpointing for HPO trials
                "enable_progress_bar": True, # Can be False for cleaner logs
                "enable_model_summary": False,
            })
        )
        return hpo_config