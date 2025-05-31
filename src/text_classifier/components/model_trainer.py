
import os
import torch
import mlflow
import joblib
import mlflow.pytorch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from torch.serialization import safe_globals
from transformers.tokenization_utils_base import BatchEncoding
import pandas as pd
import numpy as np
from text_classifier import logger
from text_classifier.entity.config_entity import ModelTrainerConfig
from text_classifier.models import LSTMClassifier, BERTClassifier, RoBERTaClassifier

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings['input_ids'])

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    def get_model(self, model_name, vocab_size=None):
        """Get model based on model name"""
        if model_name.lower() == 'lstm':
            if vocab_size is None:
                raise ValueError("vocab_size required for LSTM model")
            return LSTMClassifier(
                vocab_size=vocab_size,
                learning_rate=self.config.learning_rate
            )
        elif model_name.lower() == 'bert':
            return BERTClassifier(
                model_name='bert-base-uncased',
                learning_rate=self.config.learning_rate
            )
        elif model_name.lower() == 'roberta':
            return RoBERTaClassifier(
                model_name='roberta-base',
                learning_rate=self.config.learning_rate
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def prepare_data_loaders(self, model_name):
        """Prepare data loaders for training"""
        if model_name.lower() == 'lstm':
            # For LSTM, we need to create vocabulary and convert text to indices
            train_df = pd.read_csv(os.path.join(self.config.data_path, "train.csv"))
            val_df = pd.read_csv(os.path.join(self.config.data_path, "val.csv"))
            test_df = pd.read_csv(os.path.join(self.config.data_path, "test.csv"))
            
            # Create vocabulary
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(max_features=10000, lowercase=True)
            vectorizer.fit(train_df['text'])

            # Save the fitted vectorizer
            vectorizer_path = self.config.root_dir + "/lstm/vectorizer.pkl"
            joblib.dump(vectorizer, vectorizer_path)



            
            # Convert text to sequences
            def text_to_sequence(texts, vectorizer, max_length=512):
                sequences = []
                vocab = vectorizer.vocabulary_
                for text in texts:
                    words = text.split()
                    seq = [vocab.get(word, 0) for word in words]
                    if len(seq) > max_length:
                        seq = seq[:max_length]
                    else:
                        seq.extend([0] * (max_length - len(seq)))
                    sequences.append(seq)
                return torch.tensor(sequences, dtype=torch.long)
            
            train_sequences = text_to_sequence(train_df['text'], vectorizer)
            val_sequences = text_to_sequence(val_df['text'], vectorizer)
            test_sequences = text_to_sequence(test_df['text'], vectorizer)
            
            train_labels = torch.tensor(train_df['generated'].values, dtype=torch.long)
            val_labels = torch.tensor(val_df['generated'].values, dtype=torch.long)
            test_labels = torch.tensor(test_df['generated'].values, dtype=torch.long)
            
            # Create attention masks (all 1s for LSTM)
            train_attention = torch.ones_like(train_sequences)
            val_attention = torch.ones_like(val_sequences)
            test_attention = torch.ones_like(test_sequences)
            
            train_encodings = {
                'input_ids': train_sequences,
                'attention_mask': train_attention,
                'labels': train_labels
            }
            val_encodings = {
                'input_ids': val_sequences,
                'attention_mask': val_attention,
                'labels': val_labels
            }
            test_encodings = {
                'input_ids': test_sequences,
                'attention_mask': test_attention,
                'labels': test_labels
            }
            
            vocab_size = len(vectorizer.vocabulary_) + 1
            
        else:
            with safe_globals([BatchEncoding]):
                # For BERT/RoBERTa, load pre-tokenized data
                train_encodings = torch.load(os.path.join(self.config.data_path, "train_encodings.pt"), weights_only=False)
                val_encodings = torch.load(os.path.join(self.config.data_path, "val_encodings.pt"), weights_only=False)
                test_encodings = torch.load(os.path.join(self.config.data_path, "test_encodings.pt"), weights_only=False)
            vocab_size = None
        
        # Create datasets
        train_dataset = TextDataset(train_encodings)
        val_dataset = TextDataset(val_encodings)
        test_dataset = TextDataset(test_encodings)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.per_device_train_batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.per_device_eval_batch_size
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.per_device_eval_batch_size
        )
        
        return train_loader, val_loader, test_loader, vocab_size
    
    def train_model(self, model_name):
        """Train a specific model"""
        logger.info(f"Starting training for {model_name}")
        
        # Set up MLflow experiment
        mlflow.set_experiment(f"text_classification_{model_name}")
        
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Log parameters
            mlflow.log_params({
                'model_name': model_name,
                'learning_rate': self.config.learning_rate,
                'num_epochs': self.config.num_train_epochs,
                'batch_size': self.config.per_device_train_batch_size,
                'weight_decay': self.config.weight_decay
            })
            
            # Prepare data
            train_loader, val_loader, test_loader, vocab_size = self.prepare_data_loaders(model_name)
            
            # Get model
            model = self.get_model(model_name, vocab_size)
            
            # Set up MLflow logger
            mlf_logger = MLFlowLogger(
                experiment_name=f"text_classification_{model_name}",
                tracking_uri="mlruns"                                   # os.getenv("MLFLOW_TRACKING_URI") when use docker compose
            )
            
            # Set up callbacks
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(self.config.root_dir, model_name),
                filename=f'{model_name}-{{epoch:02d}}-{{val_acc:.2f}}',
                save_top_k=1,
                verbose=True,
                monitor='val_acc',
                mode='max'
            )
            
            early_stop_callback = EarlyStopping(
                monitor='val_acc',
                min_delta=0.001,
                patience=3,
                verbose=False,
                mode='max'
            )
            
            # Set up trainer
            trainer = pl.Trainer(
                max_epochs=self.config.num_train_epochs,
                logger=mlf_logger,
                callbacks=[checkpoint_callback, early_stop_callback],
                accelerator='auto',
                devices= 1,                                           #1 if torch.cuda.is_available() else None,
                log_every_n_steps=self.config.logging_steps,
                val_check_interval=self.config.eval_steps if self.config.eval_steps else 1.0
            )
            
            # Train model
            trainer.fit(model, train_loader, val_loader)
            
            # Test model
            test_results = trainer.test(model, test_loader, ckpt_path='best')
            
            # Log metrics
            mlflow.log_metrics({
                'test_accuracy': test_results[0]['test_acc'],
                'test_precision': test_results[0]['test_precision'],
                'test_recall': test_results[0]['test_recall'],
                'test_f1': test_results[0]['test_f1']
            })
            
            # Save model
            model_path = os.path.join(self.config.root_dir, model_name, "model")
            trainer.save_checkpoint(model_path + ".ckpt")
            
            # Log model to MLflow
            mlflow.pytorch.log_model(model, f"{model_name}_model")
            
            logger.info(f"Training completed for {model_name}")
            logger.info(f"Test results: {test_results[0]}")
            
            return test_results[0], model_path + ".ckpt"

    def train_all_models(self):
        """Train all models and compare results"""
        model_names = ['lstm', 'bert', 'roberta']
        results = {}
        model_paths = {}
        
        for model_name in model_names:
            try:
                test_result, model_path = self.train_model(model_name)
                results[model_name] = test_result
                model_paths[model_name] = model_path
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = None
                model_paths[model_name] = None
        
        # Find best model
        best_model = None
        best_accuracy = 0
        
        for model_name, result in results.items():
            if result and result['test_acc'] > best_accuracy:
                best_accuracy = result['test_acc']
                best_model = model_name
        
        logger.info(f"Best model: {best_model} with accuracy: {best_accuracy}")
        logger.info("All results:")
        for model_name, result in results.items():
            if result:
                logger.info(f"{model_name}: Acc={result['test_acc']:.4f}, "
                           f"Precision={result['test_precision']:.4f}, "
                           f"Recall={result['test_recall']:.4f}, "
                           f"F1={result['test_f1']:.4f}")
        
        # Save comparison results
        comparison_results = {
            'results': results,
            'model_paths': model_paths,
            'best_model': best_model,
            'best_accuracy': best_accuracy
        }
        
        import json
        with open(os.path.join(self.config.root_dir, "model_comparison.json"), 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        return comparison_results