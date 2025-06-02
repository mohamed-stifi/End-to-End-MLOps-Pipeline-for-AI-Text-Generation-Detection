import os
import re
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer
from pydantic import BaseModel
import torch
import pandas as pd
import joblib
from pathlib import Path
import json
import uvicorn
from typing import Optional, Dict, Any, List
from fastapi.middleware.cors import CORSMiddleware
from text_classifier.utils.kafka_producer import task_producer, close_kafka_producer # Import Kafka producer
from text_classifier.constants import KAFKA_TASKS_TOPIC # Import topic name


# Add src to Python path to allow direct imports
# This assumes the API is run from the project root directory.
# If deploying differently, this path might need adjustment or packaging.
# PROJECT_ROOT_API = Path(__file__).resolve().parent.parent
# sys.path.append(str(PROJECT_ROOT_API))
# sys.path.append(str(PROJECT_ROOT_API / "src"))


from text_classifier import logger # Global logger from __init__
# from text_classifier.constants import API_HOST, API_PORT, DEFAULT_MAX_LENGTH
from text_classifier.constants import API_HOST, API_PORT, MAX_LENGTH as DEFAULT_MAX_LENGTH
from text_classifier.models import LSTMClassifier, BERTClassifier, RoBERTaClassifier
from text_classifier.components.data_transformation import DataTransformation # For preprocessing
from text_classifier.config.configuration import ConfigurationManager # To load configs
# from main import run_pipeline # Import the pipeline runner from project's main.py


# --- Pydantic Models ---
class TextIn(BaseModel):
    text: str
    model_type: Optional[str] = None # User can specify 'lstm', 'bert', 'roberta', or None for best

class PredictionOut(BaseModel):
    prediction: str # "Human" or "AI Generated"
    label: int      # 0 for Human, 1 for AI
    confidence_score: Optional[float] = None # If model provides probabilities
    model_used: str

class RetrainResponse(BaseModel):
    message: str
    status: str

class StatsResponse(BaseModel):
    best_model_name: Optional[str] = None
    best_model_accuracy: Optional[float] = None
    best_model_f1_score: Optional[float] = None
    evaluation_timestamp: Optional[str] = None
    available_models: List[str] = []
    detailed_metrics: Optional[Dict[str, Any]] = None

class TaskSubmissionResponse(BaseModel):
    message: str
    task_type: str
    status: str
# --- FastAPI App Initialization ---
app = FastAPI(
    title="Human vs AI Text Classifier API",
    description="API for classifying text as human-written or AI-generated, and managing the MLOps pipeline.",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",  # Your frontend development server
    "http://127.0.0.1:3000",
    # Add other origins if needed (e.g., production frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Origins that are allowed to make requests
    allow_credentials=True,    # Whether to support cookies for cross-origin requests
    allow_methods=["*"],       # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],       # Allows all headers
)

# --- Global Variables & Helper Functions ---
# These will be loaded at startup
loaded_models_cache = {} # Cache for {model_type: {'model': model, 'tokenizer_or_vectorizer': obj, 'preprocessor': obj}}
api_config_manager: Optional[ConfigurationManager] = None
data_transformation_for_api: Optional[DataTransformation] = None
evaluation_report_path: Optional[Path] = None
training_summary_path: Optional[Path] = None

def get_main_project_dir():
    # Assuming api/main.py, so two parents up is the project root
    return Path(__file__).resolve().parent.parent

def initialize_api_components():
    global api_config_manager, data_transformation_for_api, evaluation_report_path, training_summary_path
    
    project_root = get_main_project_dir()
    config_filepath = project_root / "config" / "config.yaml"
    params_filepath = project_root / "config" / "params.yaml"

    if not config_filepath.exists() or not params_filepath.exists():
        logger.error("API Error: config.yaml or params.yaml not found. Cannot initialize.")
        raise RuntimeError("API configuration files missing.")

    api_config_manager = ConfigurationManager(config_filepath=config_filepath, params_filepath=params_filepath)
    
    # Get data transformation config to initialize preprocessor
    dt_config = api_config_manager.get_data_transformation_config()
    # Initialize DataTransformation with only necessary parts for preprocessing text for API
    # We create a 'dummy' config for this specific use case if some parts are not needed
    class APIDataTransformationConfig: # Simplified config for API preprocessing
        def __init__(self, tokenizer_name):
            self.tokenizer_name = tokenizer_name # Only needed for tokenizer init if we were to use its tokenizer.
                                                # But for API, we might load tokenizers per model.
                                                # The main use here is for text cleaning methods.

    # Initialize DataTransformation with a valid tokenizer name, even if not used for all models.
    # The text cleaning methods (punc, stopwords) do not depend on this HuggingFace tokenizer.
    data_transformation_for_api = DataTransformation(config=dt_config) # Pass the full dt_config

    # Paths to reports for model loading and stats
    model_eval_config = api_config_manager.get_model_evaluation_config()
    evaluation_report_path = Path(model_eval_config.metric_file_name)
    
    model_trainer_config = api_config_manager.get_model_trainer_config()
    training_summary_path = Path(model_trainer_config.root_dir) / "model_comparison.json"

    logger.info("API components initialized.")


def load_model_for_api(model_type: str):
    """Loads a specified model, its tokenizer/vectorizer, and preprocessor into cache."""
    if model_type in loaded_models_cache:
        return loaded_models_cache[model_type]

    if not api_config_manager or not training_summary_path or not data_transformation_for_api:
        raise HTTPException(status_code=503, detail="API components not initialized. Cannot load model.")

    if not training_summary_path.exists():
        raise HTTPException(status_code=404, detail=f"Training summary '{training_summary_path.name}' not found. Train models first.")
    
    with open(training_summary_path, 'r') as f:
        training_summary = json.load(f)
    
    ckpt_path_str = training_summary.get('model_paths', {}).get(model_type)
    if not ckpt_path_str or not Path(ckpt_path_str).exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint for model type '{model_type}' not found.")

    model_artifacts_root = api_config_manager.get_model_trainer_config().root_dir
    model_specific_dir = Path(model_artifacts_root) / model_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load PyTorch Lightning model from checkpoint
    try:
        if model_type == 'lstm':
            model = LSTMClassifier.load_from_checkpoint(ckpt_path_str, map_location=device)
            vectorizer_path = Path(model_specific_dir) / "vectorizer.pkl"
            if not vectorizer_path.exists():
                raise FileNotFoundError(f"LSTM vectorizer not found at {vectorizer_path}")
            tokenizer_or_vectorizer = joblib.load(vectorizer_path)
        elif model_type == 'bert':
            model = BERTClassifier.load_from_checkpoint(ckpt_path_str, map_location=device)
            # Tokenizer for BERT/RoBERTa is saved during data_transformation stage
            tokenizer_path = Path(api_config_manager.get_data_transformation_config().root_dir) / "tokenizer"
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"BERT/RoBERTa tokenizer not found at {tokenizer_path}")
            tokenizer_or_vectorizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        elif model_type == 'roberta':
            model = RoBERTaClassifier.load_from_checkpoint(ckpt_path_str, map_location=device)
            tokenizer_path = Path(api_config_manager.get_data_transformation_config().root_dir) / "tokenizer"
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"BERT/RoBERTa tokenizer not found at {tokenizer_path}")
            tokenizer_or_vectorizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.eval() # Set to eval mode
        model.to(device)

        loaded_models_cache[model_type] = {
            'model': model,
            'tokenizer_or_vectorizer': tokenizer_or_vectorizer,
            'preprocessor': data_transformation_for_api, # Re-use the initialized preprocessor for text cleaning
            'device': device
        }
        logger.info(f"Model '{model_type}' loaded successfully into cache.")
        return loaded_models_cache[model_type]
        
    except Exception as e:
        logger.error(f"Failed to load model {model_type}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not load model '{model_type}': {str(e)}")


def preprocess_input_text_for_api(text: str, model_type: str, loaded_components: dict) -> torch.Tensor:
    """Preprocesses raw text for prediction using the specified model's requirements."""
    preprocessor: DataTransformation = loaded_components['preprocessor']
    
    # 1. Basic text cleaning (punctuation, stopwords, lowercase, whitespace)
    # Create a temporary DataFrame-like structure for preprocessor methods
    temp_df = pd.DataFrame({'text': [text]})
    
    temp_df['text'] = temp_df['text'].apply(preprocessor.remove_punc)
    # Note: process_in_batches expects a Series. Apply directly for single text.
    cleaned_text_after_stopwords = preprocessor.remove_stopwords_fuzzy_optimized(temp_df['text'].iloc[0], preprocessor.stop_words)
    cleaned_text_lower = cleaned_text_after_stopwords.lower()
    final_cleaned_text = re.sub(r'\s+', ' ', cleaned_text_lower.strip()) # cleaned_text_lower.strip().replace(r'\s+', ' ', regex=True)

    if not final_cleaned_text: # If text becomes empty after cleaning
        raise ValueError("Input text became empty after preprocessing.")

    # 2. Tokenization specific to model type
    if model_type == 'lstm':
        vectorizer = loaded_components['tokenizer_or_vectorizer']
        # LSTM expects sequence of token IDs, padded
        max_len = loaded_components['model'].hparams.get('max_length', DEFAULT_MAX_LENGTH) # Use model's max_length if available

        words = final_cleaned_text.split()
        seq = [vectorizer.vocabulary_.get(word, 0) for word in words if word in vectorizer.vocabulary_]
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq.extend([0] * (max_len - len(seq))) # Assuming 0 is padding_idx
        
        input_ids = torch.tensor([seq], dtype=torch.long) # Batch size of 1
        attention_mask = (input_ids != 0).long()
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    elif model_type in ['bert', 'roberta']:
        tokenizer = loaded_components['tokenizer_or_vectorizer']
        max_len = loaded_components['model'].hparams.get('max_length', DEFAULT_MAX_LENGTH) # Or from bert_config/roberta_config

        encoded_input = tokenizer(
            final_cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt'
        )
        return encoded_input # This is a dict: {'input_ids': ..., 'attention_mask': ...}
    else:
        raise ValueError(f"Preprocessing not defined for model type: {model_type}")


@app.on_event("startup")
async def startup_event():
    """Load necessary components when API starts."""
    logger.info("FastAPI application startup...")
    try:
        initialize_api_components()
        # Optionally, pre-load the "best" model or all models here if memory allows
        # For now, models are loaded on first predict request for that model type.
        logger.info("Startup event: API components initialized.")
    except Exception as e:
        logger.error(f"API Startup failed: {e}", exc_info=True)
        # Depending on severity, might want to prevent API from starting.
        # FastAPI doesn't have a direct way to stop startup from here,
        # but subsequent requests relying on these components will fail.


# --- API Endpoints ---
@app.post("/predict", response_model=PredictionOut)
async def predict_text(text_in: TextIn):
    """
    Predicts if the input text is human-written or AI-generated.
    User can optionally specify `model_type` ('lstm', 'bert', 'roberta').
    If `model_type` is not provided, the API will attempt to use the 'best' model
    identified in the evaluation report.
    """
    if not text_in.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    model_to_use = text_in.model_type
    if not model_to_use:
        # Determine best model from evaluation report
        if not evaluation_report_path or not evaluation_report_path.exists():
            raise HTTPException(status_code=503, detail="Evaluation report not found. Cannot determine best model.")
        try:
            with open(evaluation_report_path, 'r') as f:
                eval_report = json.load(f)
            model_to_use = eval_report.get('evaluation_summary', {}).get('best_model', '') #.get('name')
            if not model_to_use:
                # Fallback if best model name is not in report (e.g. use first available from training summary)
                if training_summary_path and training_summary_path.exists():
                    with open(training_summary_path, 'r') as ts_f:
                        train_sum = json.load(ts_f)
                    available_models = list(train_sum.get('model_checkpoint_paths', {}).keys())
                    if available_models:
                        model_to_use = available_models[0] # Default to first trained model
                        logger.warning(f"Best model not identified in eval report, defaulting to first available: {model_to_use}")
                    else:
                         raise HTTPException(status_code=503, detail="No best model specified and no trained models found.")
                else:
                    raise HTTPException(status_code=503, detail="Cannot determine best model and training summary missing.")
            logger.info(f"No model_type specified by user, using best model: {model_to_use}")
        except Exception as e:
            logger.error(f"Error reading evaluation report to determine best model: {e}")
            raise HTTPException(status_code=500, detail="Error determining best model.")

    model_type = model_to_use.lower()
    
    try:
        loaded_components = load_model_for_api(model_type)
        model = loaded_components['model']
        device = loaded_components['device']

        processed_input = preprocess_input_text_for_api(text_in.text, model_type, loaded_components)
        
        # Move processed input tensors to the correct device
        input_ids = processed_input['input_ids'].to(device)
        attention_mask = processed_input.get('attention_mask').to(device) if processed_input.get('attention_mask') is not None else None
        token_type_ids = processed_input.get('token_type_ids').to(device) if processed_input.get('token_type_ids') is not None else None

        with torch.no_grad():
            if model_type == 'roberta':
                logits = model(input_ids, attention_mask)
            elif model_type == 'bert':
                 logits = model(input_ids, attention_mask, token_type_ids)
            elif model_type == 'lstm':
                logits = model(input_ids, attention_mask) # LSTM model may use attention_mask for lengths
            else: # Should have been caught by load_model_for_api
                 raise ValueError(f"Prediction logic not defined for model type: {model_type}")

        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_label_tensor = torch.max(probabilities, dim=1)
        
        predicted_label = predicted_label_tensor.item() # 0 or 1
        prediction_text = "AI Generated" if predicted_label == 1 else "Human"
        
        return PredictionOut(
            prediction=prediction_text,
            label=predicted_label,
            confidence_score=confidence.item(),
            model_used=model_type
        )

    except FileNotFoundError as e: # Specific handling for missing files
        logger.error(f"Prediction error due to missing file: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=f"A required model file was not found: {e}")
    except ValueError as e: # Specific handling for value errors (e.g. empty text after processing)
        logger.error(f"Prediction error due to invalid value: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid input or processing error: {e}")
    except Exception as e:
        logger.error(f"Error during prediction with model {model_type}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def run_training_pipeline_background():
    """Function to run the training pipeline in the background."""
    # This needs to import and run your main.py's pipeline logic.
    # Be cautious with paths and current working directory if main.py expects to be run from project root.
    # For simplicity, this is a placeholder. A robust solution might use Celery or similar task queue.
    logger.info("Background retraining task started...")
    try:
        # Assuming your main.py has a run_pipeline function:
        # Adjust path to main.py if necessary, or ensure it's callable.
        # from main import run_pipeline # This would be from the project root's main.py
        
        # Simulate running specific stages or all. For full retrain, typically 'all' or from 'training'.
        # run_pipeline(stage="all") # Or specific stages for retraining

        # Placeholder execution:
        # This simulates running the training part of the pipeline.
        # In a real scenario, you'd trigger your actual pipeline runner.
        # For example, by calling a subprocess or importing your pipeline runner.
        
        # Temporarily, let's just log this. For real execution, you'd use:
        # current_dir = os.getcwd()
        # os.chdir(PROJECT_ROOT_API) # Change to project root
        # subprocess.run([sys.executable, "main.py", "--stage", "all"], check=True)
        # os.chdir(current_dir) # Change back
        
        logger.info("Retraining pipeline (simulated) finished.")

    except Exception as e:
        logger.error(f"Background retraining task failed: {e}", exc_info=True)

'''
@app.post("/retrain", response_model=RetrainResponse)
async def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Triggers the model retraining pipeline.
    This is a simplified version that runs in the background.
    A production system would use a more robust task queue (e.g., Celery).
    """
    logger.info("Received request to trigger retraining pipeline.")
    # background_tasks.add_task(run_training_pipeline_background) # Uncomment for actual background task
    
    # For now, synchronous simulation due to potential complexity of background tasks in simple setup
    # run_training_pipeline_background() 
    
    return RetrainResponse(
        message="Retraining process has been (simulated) initiated. Check server logs for progress. In a real setup, this would be a background task.",
        status="initiated_simulation"
    )
'''
# retrain endpoint - modify to use Kafka 
@app.post("/retrain", response_model=RetrainResponse) # Original was RetrainResponse
async def trigger_retraining(): # Removed BackgroundTasks for Kafka
    logger.info("Received request to trigger retraining pipeline via Kafka.")
    
    success = task_producer.send_task(
        topic=KAFKA_TASKS_TOPIC,
        task_type="full_retrain" # Define a task type for full retraining
        # payload={} # Add any specific payload if needed
    )

    if success:
        return RetrainResponse(
            message="Full retraining task submitted to Kafka successfully. Processing will occur in the background.",
            status="submitted"
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to submit retraining task to Kafka.")


@app.get("/stats", response_model=StatsResponse)
async def get_model_stats():
    """Provides model monitoring statistics (e.g., accuracy of the best model)."""
    if not evaluation_report_path or not evaluation_report_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation report not found. Run evaluation first.")
    
    if not training_summary_path or not training_summary_path.exists():
        raise HTTPException(status_code=404, detail="Training summary not found. Run training first.")

    try:
        with open(evaluation_report_path, 'r') as f:
            eval_report = json.load(f)
        
        with open(training_summary_path, 'r') as f:
            train_summary = json.load(f)

        best_model_info = eval_report.get('evaluation_summary', {})
        best_model_name = best_model_info.get('best_model')
        
        available_model_types = list(train_summary.get('model_paths', {}).keys())

        if best_model_name and best_model_name in eval_report.get('model_results', {}):
            # best_model_metrics = eval_report['model_results'][best_model_name].get('metrics', {})
            return StatsResponse(
                best_model_name=best_model_name,
                best_model_accuracy=eval_report['model_results'][best_model_name].get('accuracy'),
                best_model_f1_score=eval_report['model_results'][best_model_name].get('f1_score'),
                evaluation_timestamp=eval_report['evaluation_summary'].get('evaluation_date'),
                available_models=available_model_types,
                detailed_metrics=eval_report['model_results'].get(best_model_name)
            )
        else: # Fallback if best model not clearly identified or missing detailed metrics
            return StatsResponse(
                best_model_name=best_model_name,
                evaluation_timestamp=eval_report['evaluation_summary'].get('evaluation_date'),
                available_models=available_model_types,
                detailed_metrics={"info": "Detailed metrics for the best model not fully available in report."}
            )
            
    except Exception as e:
        logger.error(f"Error retrieving model stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model statistics: {str(e)}")

# FastAPI Lifespan event to close Kafka producer on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down...")
    close_kafka_producer() # Close Kafka producer
    logger.info("Kafka producer closed.")

# ENDPOINT: /evaluate
@app.post("/evaluate", response_model=TaskSubmissionResponse)
async def trigger_evaluation():
    logger.info("Received request to trigger model evaluation pipeline via Kafka.")
    
    success = task_producer.send_task(
        topic=KAFKA_TASKS_TOPIC,
        task_type="evaluate_models"
        # payload={} # Add any specific payload if needed
    )

    if success:
        return TaskSubmissionResponse(
            message="Model evaluation task submitted to Kafka successfully. Processing will occur in the background.",
            task_type="evaluate_models",
            status="submitted"
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to submit model evaluation task to Kafka.")


# ENDPOINT: /parameter_tuning
@app.post("/parameter_tuning", response_model=TaskSubmissionResponse)
async def trigger_parameter_tuning():
    logger.info("Received request to trigger hyperparameter optimization pipeline via Kafka.")

    success = task_producer.send_task(
        topic=KAFKA_TASKS_TOPIC,
        task_type="tune_hyperparameters"
        # payload={} # Add any specific payload if needed, e.g., specific models to tune
    )

    if success:
        return TaskSubmissionResponse(
            message="Hyperparameter optimization task submitted to Kafka successfully. Processing will occur in the background.",
            task_type="tune_hyperparameters",
            status="submitted"
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to submit hyperparameter optimization task to Kafka.")


if __name__ == "__main__":
    # This allows running the API directly using `python api/main.py`
    # Ensure PROJECT_ROOT_API and sys.path modifications are effective.
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    
    # Check if essential config files exist before trying to run uvicorn
    project_root = get_main_project_dir()
    config_file = project_root / "config" / "config.yaml"
    params_file = project_root / "config" / "params.yaml"

    if not config_file.exists() or not params_file.exists():
        print(f"ERROR: Missing configuration files. Ensure 'config/config.yaml' and 'config/params.yaml' exist at {project_root}.", file=sys.stderr)
        sys.exit(1)
        
    uvicorn.run(app, host=API_HOST, port=API_PORT) # reload=True for development