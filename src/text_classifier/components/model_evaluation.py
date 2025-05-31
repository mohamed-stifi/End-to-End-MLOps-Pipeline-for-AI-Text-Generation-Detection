import os
import json
import mlflow
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from text_classifier import logger
from text_classifier.entity.config_entity import ModelEvaluationConfig
from text_classifier.models import LSTMClassifier, BERTClassifier, RoBERTaClassifier

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_model(self, model_path, model_type):
        """Load trained model from checkpoint"""
        try:
            if model_type.lower() == 'lstm':
                model = LSTMClassifier.load_from_checkpoint(model_path)
            elif model_type.lower() == 'bert':
                model = BERTClassifier.load_from_checkpoint(model_path)
            elif model_type.lower() == 'roberta':
                model = RoBERTaClassifier.load_from_checkpoint(model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_type}: {e}")
            return None

    def evaluate_model(self, model, test_loader, model_name):
        """Evaluate a single model"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask', None)
                labels = batch['labels']
                
                # Get predictions
                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Classification report
        class_report = classification_report(all_labels, all_preds, 
                                           target_names=['Human', 'AI Generated'])
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs
        }
        
        return results

    def plot_confusion_matrix(self, cm, model_name, save_path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Human', 'AI Generated'],
                    yticklabels=['Human', 'AI Generated'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_model_comparison(self, results_dict, save_path):
        """Plot model comparison chart"""
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [results_dict[model][metric] for model in models]
            axes[i].bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral'])
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def generate_evaluation_report(self, results_dict):
        """Generate comprehensive evaluation report"""
        report = {
            'evaluation_summary': {
                'total_models_evaluated': len(results_dict),
                'evaluation_date': pd.Timestamp.now().isoformat(),
                'best_model': None,
                'best_accuracy': 0
            },
            'model_results': results_dict,
            'model_ranking': []
        }
        
        # Find best model and create ranking
        model_scores = []
        for model_name, results in results_dict.items():
            if results:
                score = (results['accuracy'] + results['f1_score']) / 2  # Combined score
                model_scores.append((model_name, score, results['accuracy']))
                
                if results['accuracy'] > report['evaluation_summary']['best_accuracy']:
                    report['evaluation_summary']['best_model'] = model_name
                    report['evaluation_summary']['best_accuracy'] = results['accuracy']
        
        # Sort by combined score
        model_scores.sort(key=lambda x: x[1], reverse=True)
        report['model_ranking'] = [
            {
                'rank': i+1,
                'model': score[0],
                'combined_score': score[1],
                'accuracy': score[2]
            } for i, score in enumerate(model_scores)
        ]
        
        return report

    def evaluate_all_models(self, data_loader_fun):
        """Evaluate all trained models"""
        logger.info("Starting model evaluation...")
        
        # Load model comparison results
        comparison_file = self.config.comparison_file # os.path.join(self.config.data_path, "model_comparison.json")
        
        if not os.path.exists(comparison_file):
            logger.error(f"Model comparison file {comparison_file } not found. Please train models first.")
            return None
        
        with open(comparison_file, 'r') as f:
            comparison_data = json.load(f)
        
        model_paths = comparison_data['model_paths']
        results_dict = {}
        
        # Load test data
        '''from text_classifier.components.model_trainer import TextDataset
        from torch.utils.data import DataLoader
        
        test_encodings = torch.load(os.path.join(self.config.data_path, "test_encodings.pt"))
        test_dataset = TextDataset(test_encodings)
        test_loader = DataLoader(test_dataset, batch_size=32)'''
        
        # Evaluate each model
        for model_name, model_path in model_paths.items():
            if model_path and os.path.exists(model_path):
                logger.info(f"Evaluating {model_name}...")
                
                try:
                    train_loader, val_loader, test_loader, vocab_size = data_loader_fun(model_name)
                    model = self.load_model(model_path, model_name)
                    if model:
                        results = self.evaluate_model(model, test_loader, model_name)
                        results_dict[model_name] = results
                        
                        # Plot confusion matrix
                        cm_path = os.path.join(self.config.root_dir, f"{model_name}_confusion_matrix.png")
                        self.plot_confusion_matrix(
                            np.array(results['confusion_matrix']), 
                            model_name, 
                            cm_path
                        )
                        
                        logger.info(f"{model_name} evaluation completed")
                        logger.info(f"Accuracy: {results['accuracy']:.4f}")
                        
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
                    results_dict[model_name] = None
            else:
                logger.warning(f"Model path not found for {model_name}")
                results_dict[model_name] = None
        
        # Generate comparison plots
        if results_dict:
            # Filter out None results
            valid_results = {k: v for k, v in results_dict.items() if v is not None}
            
            if valid_results:
                comparison_plot_path = os.path.join(self.config.root_dir, "model_comparison.png")
                self.plot_model_comparison(valid_results, comparison_plot_path)
                
                # Generate evaluation report
                evaluation_report = self.generate_evaluation_report(valid_results)
                
                # Save evaluation report
                report_path = os.path.join(self.config.root_dir, "evaluation_report.json")
                with open(report_path, 'w') as f:
                    json.dump(evaluation_report, f, indent=2, default=str)
                
                # Log to MLflow
                mlflow.set_experiment("model_evaluation")
                with mlflow.start_run(run_name="model_comparison"):
                    for model_name, results in valid_results.items():
                        mlflow.log_metrics({
                            f"{model_name}_accuracy": results['accuracy'],
                            f"{model_name}_precision": results['precision'],
                            f"{model_name}_recall": results['recall'],
                            f"{model_name}_f1": results['f1_score']
                        })
                    
                    # Log artifacts
                    mlflow.log_artifact(comparison_plot_path)
                    mlflow.log_artifact(report_path)
                    for model_name in valid_results.keys():
                        cm_path = os.path.join(self.config.root_dir, f"{model_name}_confusion_matrix.png")
                        if os.path.exists(cm_path):
                            mlflow.log_artifact(cm_path)
                
                logger.info(f"Evaluation completed. Best model: {evaluation_report['evaluation_summary']['best_model']}")
                return evaluation_report
            
        logger.warning("No valid model results found")
        return None
