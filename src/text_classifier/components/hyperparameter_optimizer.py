import os
import json
import optuna
import mlflow
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback


from text_classifier import logger
from text_classifier.entity.config_entity import HyperparameterOptimizationConfig
from text_classifier.models import LSTMClassifier, BERTClassifier, RoBERTaClassifier
from text_classifier.components.model_trainer import TextDataset # Re-use Dataset
from text_classifier.constants import MAX_LENGTH as DEFAULT_MAX_LENGTH
from text_classifier.utils.common import save_json, create_directories
import joblib # For LSTM vectorizer

class HyperparameterOptimizer:
    def __init__(self, config: HyperparameterOptimizationConfig):
        self.config = config
        create_directories([self.config.root_dir])
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

    def _get_model_instance(self, model_type: str, trial: optuna.Trial, vocab_size_lstm: Optional[int] = None) -> pl.LightningModule:
        """Instantiates a model with hyperparameters suggested by Optuna."""
        model_type = model_type.lower()
        param_ranges = self.config.hpo_params_ranges.get(model_type, {})
        
        def suggest_param(param_name, config):
            if param_name not in config: # If param not in HPO ranges, use default from main config or model default
                return None # Let model use its default
            
            p_type = config[param_name]['type']
            if p_type == "float":
                return trial.suggest_float(param_name, config[param_name]['low'], config[param_name]['high'], log=config[param_name].get('log', False))
            elif p_type == "int":
                return trial.suggest_int(param_name, config[param_name]['low'], config[param_name]['high'])
            elif p_type == "categorical":
                return trial.suggest_categorical(param_name, config[param_name]['choices'])
            return None

        if model_type == "lstm":
            print(param_ranges)
            if vocab_size_lstm is None:
                raise ValueError("vocab_size_lstm required for LSTM HPO.")
            return LSTMClassifier(
                vocab_size=vocab_size_lstm,
                embedding_dim=suggest_param("embedding_dim", param_ranges) or 128,
                hidden_dim=suggest_param("hidden_dim", param_ranges) or 256,
                num_layers=suggest_param("num_layers", param_ranges) or 2,
                dropout=suggest_param("dropout", param_ranges) or 0.3,
                learning_rate=suggest_param("learning_rate", param_ranges) or 1e-3
            )
        elif model_type == "bert":
            # Bert model_name is usually fixed during HPO, focus on LR, dropout etc.
            # If you want to tune bert_model_name, add it to hpo_params_ranges.
            return BERTClassifier(
                model_name=self.config.hpo_params_ranges.get(model_type, {}).get("model_name", "bert-base-uncased"), # Fixed or from params
                learning_rate=suggest_param("learning_rate", param_ranges) or 2e-5,
                dropout=suggest_param("dropout", param_ranges) or 0.1,
                # warmup_steps_ratio could also be tuned
                warmup_steps_ratio=suggest_param("warmup_steps_ratio", param_ranges) or 0.1
            )
        elif model_type == "roberta":
            return RoBERTaClassifier(
                model_name=self.config.hpo_params_ranges.get(model_type, {}).get("model_name", "roberta-base"),
                learning_rate=suggest_param("learning_rate", param_ranges) or 2e-5,
                dropout=suggest_param("dropout", param_ranges) or 0.1,
                warmup_steps_ratio=suggest_param("warmup_steps_ratio", param_ranges) or 0.1
            )
        else:
            raise ValueError(f"Unsupported model type for HPO: {model_type}")

    def _prepare_hpo_dataloaders(self, model_type: str) -> Tuple[DataLoader, DataLoader, Optional[int]]:
        """Prepares train and validation dataloaders for HPO trials."""
        model_type = model_type.lower()
        data_transform_path = self.config.data_path # Path to 'artifacts/data_transformation'
        
        train_encodings, val_encodings = {}, {}
        vocab_size_lstm = None
        
        # This logic is similar to ModelTrainer.prepare_data_loaders, simplified for HPO
        # It assumes the LSTM vectorizer is already created by a previous full training run of LSTM,
        # or that HPO for LSTM fits its own vectorizer (less ideal for direct comparison).
        # For HPO, it's better if the vectorizer is fixed.
        # Let's assume we load a pre-fitted vectorizer for LSTM HPO.
        if model_type == 'lstm':
            import pandas as pd # Local import
            # Path to where the main ModelTrainer would save the LSTM vectorizer
            # This implies HPO runs AFTER at least one full training cycle (or a dedicated vectorizer fitting step)
            # OR, HPO for LSTM might need its own simple vectorizer fitting on the HPO train split if run independently.
            # For simplicity, let's assume vectorizer is in model_trainer output.
            # This is a dependency: HPO for LSTM needs artifacts from model_trainer or similar.
            lstm_model_trainer_artifacts_dir = Path(str(self.config.data_path).replace("data_transformation", "model_trainer")) / "lstm"
            vectorizer_path = lstm_model_trainer_artifacts_dir / "vectorizer.pkl"

            if not vectorizer_path.exists():
                # Fallback: if no global vectorizer, fit one quickly for HPO (less robust)
                logger.warning(f"LSTM vectorizer not found at {vectorizer_path}. Fitting a temporary one for HPO on HPO train data.")
                from sklearn.feature_extraction.text import CountVectorizer
                train_df_hpo = pd.read_csv(data_transform_path / "train.csv") # Use the main train split for HPO as well
                train_df_hpo['text'] = train_df_hpo['text'].astype(str).fillna('')
                vectorizer = CountVectorizer(max_features=10000, lowercase=False)
                vectorizer.fit(train_df_hpo['text'])
                # We don't save this temporary vectorizer globally.
            else:
                vectorizer = joblib.load(vectorizer_path)
            
            vocab_size_lstm = len(vectorizer.vocabulary_) + 1
            
            def lstm_text_to_encoding(csv_path, max_len):
                df = pd.read_csv(csv_path)
                df['text'] = df['text'].astype(str).fillna('')
                sequences = []
                for text in df['text']:
                    words = text.split()
                    seq = [vectorizer.vocabulary_.get(word, 0) for word in words if word in vectorizer.vocabulary_]
                    if len(seq) > max_len: seq = seq[:max_len]
                    else: seq.extend([0] * (max_len - len(seq)))
                    sequences.append(seq)
                input_ids = torch.tensor(sequences, dtype=torch.long)
                labels = torch.tensor(df['generated'].values, dtype=torch.long)
                attention_mask = (input_ids != 0).long()
                return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

            max_len_lstm = self.config.hpo_params_ranges.get(model_type, {}).get("max_length", DEFAULT_MAX_LENGTH)
            train_encodings = lstm_text_to_encoding(data_transform_path / "train.csv", max_len_lstm)
            val_encodings = lstm_text_to_encoding(data_transform_path / "val.csv", max_len_lstm)

        else: # For BERT/RoBERTa
            train_encodings = torch.load(data_transform_path / "train_encodings.pt")
            val_encodings = torch.load(data_transform_path / "val_encodings.pt")

        train_dataset = TextDataset(train_encodings)
        val_dataset = TextDataset(val_encodings)
        
        # Batch size could also be a hyperparameter, but let's fix it for now from main params
        # Or define a specific hpo_batch_size in params.yaml
        hpo_batch_size = self.config.hpo_params_ranges.get(model_type, {}).get("batch_size", 32)

        train_loader = DataLoader(train_dataset, batch_size=hpo_batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=hpo_batch_size, num_workers=2)
        
        return train_loader, val_loader, vocab_size_lstm


    def _objective(self, trial: optuna.Trial, model_type: str, train_loader: DataLoader, val_loader: DataLoader, vocab_size_lstm: Optional[int]) -> float:
        """Optuna objective function for a single trial."""
        # MLflow setup for each trial
        # Each trial is a separate MLflow run, nested under the main HPO experiment/run for that model_type
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) as run:
            mlflow.log_params(trial.params) # Log Optuna suggested params
            mlflow.set_tag("optuna_study_name", trial.study.study_name)
            mlflow.set_tag("optuna_trial_number", trial.number)
            mlflow.set_tag("model_type_tuned", model_type)

            try:
                model = self._get_model_instance(model_type, trial, vocab_size_lstm)
                
                # Configure PyTorch Lightning Trainer for this trial
                trainer_args = self.config.trainer_hpo_config.copy()
                # Allow overriding trainer args per trial if specified in hpo_params_ranges
                if "pl_max_epochs" in trial.params: # Example
                    trainer_args["max_epochs"] = trial.params["pl_max_epochs"]

                # OptunaPruning callback for early stopping of unpromising trials
                # pruning_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor=self.config.metric_to_optimize)
                # pruning_callback = PyTorchLightningPruningCallback(trial, monitor=self.config.metric_to_optimize)

                # We don't need MLFlowLogger for individual trials if the parent Optuna study logs to MLflow.
                # However, having it allows detailed logging per trial.
                # If parent MLflow run is already active, this creates a nested run.
                trial_mlf_logger = MLFlowLogger(
                    experiment_name=mlflow.get_experiment_by_name(trial.study.study_name).experiment_id if trial.study.study_name else "HPO_Trials",
                    tracking_uri=self.config.mlflow_tracking_uri,
                    run_id=run.info.run_id # Link to current MLflow run for this trial
                )

                trainer = pl.Trainer(
                    logger=trial_mlf_logger, # Log metrics for this trial
                    callbacks=[ RichProgressBar(leave=True)], # Add RichProgressBar if desired
                    **trainer_args, # Unpack common trainer args
                    accelerator='auto', # 'gpu' if torch.cuda.is_available() else 'cpu',
                    devices=1,
                )

                trainer.fit(model, train_loader, val_loader)
                
                # The metric to optimize should be logged by PyTorch Lightning (e.g., 'val_acc')
                # Pruning callback uses it. We need to return it.
                # It's usually the value from the last epoch or best checkpoint if enabled.
                # If using EarlyStopping, it might be from an earlier epoch.
                optimized_metric_value = trainer.callback_metrics.get(self.config.metric_to_optimize)

                if optimized_metric_value is None:
                    logger.warning(f"Metric '{self.config.metric_to_optimize}' not found in trainer.callback_metrics for trial {trial.number}. Available: {trainer.callback_metrics.keys()}. Returning default low/high value.")
                    return float('-inf') if self.config.direction == "maximize" else float('inf')
                
                # Log the final optimized metric to MLflow for this trial explicitly
                mlflow.log_metric(f"final_{self.config.metric_to_optimize}", optimized_metric_value.item())
                
                return optimized_metric_value.item()

            except optuna.exceptions.TrialPruned as e:
                logger.info(f"Trial {trial.number} pruned: {e}")
                raise # Re-raise to let Optuna handle it
            except Exception as e:
                logger.error(f"Error in HPO trial {trial.number} for model {model_type}: {e}", exc_info=True)
                mlflow.log_param("trial_status", "failed")
                mlflow.set_tag("trial_error", str(e))
                # Return a very bad value if the trial fails, so Optuna doesn't favor it
                return float('-inf') if self.config.direction == "maximize" else float('inf')


    def optimize(self):
        """Runs HPO for all specified model types."""
        overall_best_params = {}

        for model_type in self.config.model_types_to_tune:
            logger.info(f"--- Starting Hyperparameter Optimization for model type: {model_type} ---")
            
            # Prepare data once per model_type (as it might differ for LSTM vs Transformers)
            train_loader, val_loader, vocab_size_lstm = self._prepare_hpo_dataloaders(model_type)

            # Use MLflow as a backend for Optuna to log studies and trials automatically
            # Need to set an experiment for the Optuna study itself
            study_experiment_name = f"{self.config.study_name_prefix}_{model_type}"
            mlflow.set_experiment(study_experiment_name) # Each model type gets its own HPO experiment
            
            # This callback logs study-level information and best trial parameters to MLflow.
            # It works by creating a parent MLflow run for the Optuna study.
            mlflow_optuna_callback = optuna.integration.MLflowCallback(
                tracking_uri=self.config.mlflow_tracking_uri,
                metric_name=self.config.metric_to_optimize, # Optuna will log best trial's metric under this name
                # create_experiment=False # Since we set it above
            )
            # Note: MLflowCallback creates its own parent run for the study.
            # Individual trials (_objective function) will create NESTED runs.

            study = optuna.create_study(
                study_name=f"{self.config.study_name_prefix}_{model_type}_{Path(self.config.data_path).name}", # Unique study name
                direction=self.config.direction,
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2) # Example pruner
            )
            
            study.optimize(
                lambda trial: self._objective(trial, model_type, train_loader, val_loader, vocab_size_lstm),
                n_trials=self.config.n_trials,
                callbacks=[mlflow_optuna_callback], # Add MLflow callback for Optuna study
                # timeout=3600 # Optional: timeout in seconds for the study
            )

            logger.info(f"HPO completed for {model_type}.")
            logger.info(f"Best trial for {model_type}:")
            logger.info(f"  Value ({self.config.metric_to_optimize}): {study.best_value}")
            logger.info(f"  Params: {study.best_params}")

            overall_best_params[model_type] = {
                'best_value': study.best_value,
                'best_params': study.best_params,
                'study_name': study.study_name,
                # 'mlflow_run_id_study': mlflow_optuna_callback.study_run_id # If accessible, or find via tags
            }
        
        # Save the overall HPO results to a JSON file
        hpo_summary_path = self.config.root_dir / "hpo_optimization_summary.json"
        save_json(hpo_summary_path, overall_best_params)
        logger.info(f"Overall HPO summary saved to: {hpo_summary_path}")
        
        # Log this summary as an artifact in a general "HPO_Master_Report" run if desired
        # with mlflow.start_run(run_name="HPO_Master_Report", experiment_id="<general_hpo_experiment_id>"):
        #    mlflow.log_artifact(hpo_summary_path)

        return overall_best_params