import os
import sys
import yaml
import json
import joblib
import logging
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError

logger = logging.getLogger(__name__)

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns ConfigBox type"""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def write_yaml(path_to_yaml: Path, data: Union[dict, ConfigBox]): 
    """writes data to a yaml file"""
    try:
        # If data is ConfigBox, convert it to dict for yaml.dump
        if isinstance(data, ConfigBox):
            data_to_write = data.to_dict()
        else:
            data_to_write = data

        with open(path_to_yaml, 'w') as yaml_file:
            yaml.dump(data_to_write, yaml_file, default_flow_style=False, sort_keys=False)
        logger.info(f"yaml file: {path_to_yaml} saved successfully")
    except Exception as e:
        logger.error(f"Error writing yaml to {path_to_yaml}: {e}")
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories"""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data"""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data"""
    with open(path) as f:
        content = json.load(f)
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)

def save_bin(data: Any, path: Path):
    """save binary file"""
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")

# @ensure_annotations # Removed for load_bin
def load_bin(path: Path) -> Any:
    """load binary data"""
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB"""
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# function to update params.yaml
def update_params_from_hpo(
    hpo_summary_path: Path,
    params_yaml_path: Path,
    model_config_map: Dict[str, str] # Maps HPO model type (e.g., "lstm") to params.yaml key (e.g., "LSTM")
):
    """
    Updates params.yaml with the best hyperparameters found during HPO.

    Args:
        hpo_summary_path (Path): Path to the hpo_optimization_summary.json file.
        params_yaml_path (Path): Path to the params.yaml file to be updated.
        model_config_map (Dict[str, str]): A dictionary mapping the model type key
                                           used in HPO summary (e.g., "lstm") to the
                                           corresponding top-level key in params.yaml
                                           (e.g., "LSTM").
    """
    try:
        if not hpo_summary_path.exists():
            logger.warning(f"HPO summary file not found at {hpo_summary_path}. Skipping params update.")
            return False

        hpo_summary = load_json(hpo_summary_path)
        params_data = read_yaml(params_yaml_path) # Returns ConfigBox

        updated_params_count = 0

        for model_type_hpo, hpo_results in hpo_summary.items():
            if model_type_hpo in model_config_map:
                params_key = model_config_map[model_type_hpo] # e.g., "LSTM"
                best_params_from_hpo = hpo_results.get("best_params", {})

                if not best_params_from_hpo:
                    logger.warning(f"No 'best_params' found for model type '{model_type_hpo}' in HPO summary.")
                    continue

                if params_key not in params_data:
                    logger.warning(f"Key '{params_key}' not found in params.yaml. Cannot update for model type '{model_type_hpo}'.")
                    params_data[params_key] = {} # Create the key if it doesn't exist

                logger.info(f"Updating parameters for '{params_key}' in params.yaml with HPO results...")
                for param_name, param_value in best_params_from_hpo.items():
                    # Here, we need to decide where these HPO params go.
                    # Your params.yaml has model-specific sections (LSTM, BERT, RoBERTa)
                    # and a general TrainingArguments section.
                    # HPO tunes parameters like 'learning_rate', 'dropout'.
                    # These could override values in LSTM.learning_rate or TrainingArguments.learning_rate.

                    # Strategy: Prioritize model-specific section in params.yaml.
                    if param_name in params_data[params_key]:
                        logger.info(f"  Updating {params_key}.{param_name}: {params_data[params_key][param_name]} -> {param_value}")
                        params_data[params_key][param_name] = param_value
                        updated_params_count += 1
                    # If not in model-specific, check if it's a common TrainingArgument
                    elif "TrainingArguments" in params_data and param_name in params_data.TrainingArguments:
                        # This logic is a bit tricky: should HPO's 'learning_rate' override the global one
                        # or only if the model-specific one doesn't exist?
                        # For now, let's assume HPO results are primarily for model-specific sections.
                        # If HPO'd 'learning_rate' is for a model, it should go into 'LSTM.learning_rate', not 'TrainingArguments.learning_rate'.
                        logger.warning(f"  Parameter '{param_name}' from HPO for '{model_type_hpo}' also found in 'TrainingArguments'. "
                                       f"Currently, HPO params update model-specific sections ({params_key}).")
                    else:
                        # If the param is new for this model section, add it
                        logger.info(f"  Adding new parameter {params_key}.{param_name}: {param_value}")
                        params_data[params_key][param_name] = param_value
                        updated_params_count +=1
            else:
                logger.warning(f"Model type '{model_type_hpo}' from HPO summary not found in model_config_map. Skipping.")

        if updated_params_count > 0:
            write_yaml(params_yaml_path, params_data)
            logger.info(f"Successfully updated {params_yaml_path} with {updated_params_count} parameters from HPO results.")
            return True
        else:
            logger.info("No parameters were updated in params.yaml based on HPO results.")
            return False

    except Exception as e:
        logger.error(f"Failed to update params.yaml from HPO summary: {e}", exc_info=True)
        return False
