import pytest
from pathlib import Path
import shutil
import yaml
import os
import nltk

from text_classifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from text_classifier.pipeline.stage_02_data_validation import DataValidationPipeline
from text_classifier.pipeline.stage_03_data_transformation import DataTransformationPipeline
from text_classifier.pipeline.stage_04_model_trainer import ModelTrainingPipeline
# from text_classifier.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline # Optional for basic flow
from text_classifier.config.configuration import ConfigurationManager
from text_classifier.utils.common import read_yaml, create_directories

# --- Test Configuration ---
# We need to override the default config/params to use test-specific paths and smaller settings.
# This can be done by creating temporary config files or patching ConfigurationManager.

# Path to fixture files
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
DUMMY_DATA_ZIP = FIXTURES_DIR / "dummy_data.zip" # Contains dummy_data.csv
SAMPLE_CONFIG_FILE_ORIG = FIXTURES_DIR / "sample_config.yaml"
SAMPLE_PARAMS_FILE_ORIG = FIXTURES_DIR / "sample_params.yaml"


@pytest.fixture(scope="module") # Run once per module for efficiency
def test_pipeline_environment(tmp_path_factory):
    # Create a temporary directory for all artifacts of this integration test run
    base_temp_dir = tmp_path_factory.mktemp("integration_test_run")
    
    # 1. Prepare temporary config files
    temp_config_dir = base_temp_dir / "config"
    temp_config_dir.mkdir()
    
    temp_config_yaml = temp_config_dir / "config.yaml"
    temp_params_yaml = temp_config_dir / "params.yaml"

    # Load and modify sample_config.yaml to use base_temp_dir for artifacts_root
    # and point to the dummy data zip.
    cfg_content = read_yaml(SAMPLE_CONFIG_FILE_ORIG)
    
    # Override artifacts_root and all paths derived from it
    new_artifacts_root = base_temp_dir / "artifacts_integration"
    old_artifacts_root_placeholder = "test_artifacts" # Placeholder from sample_config.yaml

    def replace_paths_in_config(data_struct, old_base, new_base):
        if isinstance(data_struct, dict):
            return {k: replace_paths_in_config(v, old_base, new_base) for k, v in data_struct.items()}
        elif isinstance(data_struct, list):
            return [replace_paths_in_config(i, old_base, new_base) for i in data_struct]
        elif isinstance(data_struct, str) and old_base in data_struct:
            return data_struct.replace(old_base, str(new_base))
        return data_struct

    cfg_content = replace_paths_in_config(cfg_content, old_artifacts_root_placeholder, new_artifacts_root)
    cfg_content['artifacts_root'] = str(new_artifacts_root) # Explicitly set root

    # Override data ingestion source to use local dummy zip
    cfg_content['data_ingestion']['source_URL'] = f"file://{DUMMY_DATA_ZIP.resolve()}"
    # Ensure local_data_file path is also under the new artifacts root
    cfg_content['data_ingestion']['local_data_file'] = str(new_artifacts_root / "data_ingestion" / "data.zip")
    
    with open(temp_config_yaml, 'w') as f:
        yaml.dump(cfg_content, f)

    # Copy sample_params.yaml (it's already set for small models/epochs)
    shutil.copy(SAMPLE_PARAMS_FILE_ORIG, temp_params_yaml)

    # 2. Patch ConfigurationManager to use these temporary config files
    original_config_manager_init = ConfigurationManager.__init__
    
    def patched_init(self, config_filepath=None, params_filepath=None):
        # If called by test, use temp files. Otherwise, default behavior.
        # This check isn't perfect, better to pass explicitly if pipeline stages allow.
        # For now, this global patch works if tests run isolatedly.
        nonlocal temp_config_yaml, temp_params_yaml
        print(f"Patched ConfigurationManager using: {temp_config_yaml}, {temp_params_yaml}")
        original_config_manager_init(self, config_filepath=temp_config_yaml, params_filepath=temp_params_yaml)

    ConfigurationManager.__init__ = patched_init
    
    # Yield the path to new_artifacts_root so tests can inspect it
    yield str(new_artifacts_root)

    # Teardown: Restore original ConfigurationManager init
    ConfigurationManager.__init__ = original_config_manager_init
    # tmp_path_factory handles cleanup of base_temp_dir


@pytest.mark.slow # Mark as slow as it involves file I/O and model loading
def test_full_pipeline_flow_simplified(test_pipeline_environment, caplog):
    # test_pipeline_environment fixture has already set up configs and patched ConfigurationManager
    # caplog fixture captures logging output
    import logging
    caplog.set_level(logging.INFO)

    artifacts_root = Path(test_pipeline_environment)
    
    # --- Stage 1: Data Ingestion ---
    print("\n--- Running Data Ingestion ---")
    ingestion_pipeline = DataIngestionPipeline()
    ingestion_pipeline.main()
    
    # Assertions for Data Ingestion
    # Check if dummy_data.zip was "downloaded" (copied) and extracted
    # Path from our modified sample_config.yaml
    mgr_for_paths = ConfigurationManager() # Uses patched init
    di_conf = mgr_for_paths.get_data_ingestion_config()

    assert (Path(di_conf.local_data_file)).exists()
    extracted_csv_path = Path(di_conf.unzip_dir) / di_conf.main_csv_file_name
    assert extracted_csv_path.exists()
    print(f"Data Ingestion successful. Extracted CSV: {extracted_csv_path}")

    # --- Stage 2: Data Validation ---
    print("\n--- Running Data Validation ---")
    validation_pipeline = DataValidationPipeline()
    validation_pipeline.main()
    
    # Assertions for Data Validation
    dv_conf = mgr_for_paths.get_data_validation_config()
    status_file = Path(dv_conf.status_file)
    assert status_file.exists()
    with open(status_file, 'r') as f:
        status_content = f.read()
        assert "Validation status: True" in status_content
    print("Data Validation successful.")

    # --- Stage 3: Data Transformation ---
    print("\n--- Running Data Transformation ---")
    transformation_pipeline = DataTransformationPipeline()
    # This stage needs nltk data. Mock it if it becomes an issue in CI.
    # For now, assume it's available or test_data_transformation.py handles mocks.
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    transformation_pipeline.main()
    
    # Assertions for Data Transformation
    dt_conf = mgr_for_paths.get_data_transformation_config()
    assert (Path(dt_conf.root_dir) / "train.csv").exists()
    assert (Path(dt_conf.root_dir) / "train_encodings.pt").exists() # If BERT/RoBERTa like tokenizer
    assert (Path(dt_conf.root_dir) / "tokenizer").exists() # Tokenizer saved
    print("Data Transformation successful.")

    # --- Stage 4: Model Training (Simplified: one model, few epochs) ---
    # The sample_params.yaml should be configured for a tiny model and 1 epoch.
    # The `ModelTrainerConfig` in `sample_config.yaml` might pick a specific model,
    # or `ModelTrainer.train_all_models` will run based on available types.
    # Let's assume `train_all_models_and_compare` is called.
    print("\n--- Running Model Training ---")
    # ModelTrainingPipeline needs mlflow_tracking_uri. sample_config.yaml has it.
    # Ensure MLflow doesn't try to connect to a server if not intended for test.
    # 'file:///tmp/test_mlruns' from sample_config.yaml is fine.
    
    # Make sure the MLflow tracking URI directory exists
    os.makedirs(mgr_for_paths.config.mlflow_tracking_uri.replace("file://", ""), exist_ok=True)

    training_pipeline = ModelTrainingPipeline()
    training_pipeline.main()

    # Assertions for Model Training
    mt_conf = mgr_for_paths.get_model_trainer_config()
    # Check if model artifacts are created for at least one model type (e.g., lstm from sample_params)
    # Based on `sample_params.yaml` and `sample_config.yaml`, only LSTM might be trained if `model_types_to_tune` from HPO config influences this.
    # Or, if ModelTrainerConfig.model_name is "bert-base-uncased", it might try that.
    # Let's assume it trains models like 'lstm', 'bert' as per the original setup.
    # The sample_params uses "prajjwal1/bert-tiny" via TOKENIZER_NAME, affecting BERT/RoBERTa.
    
    # Check for model_comparison.json
    comparison_file = Path(mt_conf.root_dir) / "model_comparison.json" # Updated name if stage_04 changed
    if not comparison_file.exists():
         comparison_file = Path(mt_conf.root_dir) / "model_training_comparison.json" # Check alternative name

    assert comparison_file.exists(), f"Comparison file not found at {comparison_file} or alternative."
    
    # Check for at least one model directory (e.g., lstm or bert)
    # The actual models trained depend on constants.MODEL_TYPES and logic in ModelTrainer
    # sample_config.yaml -> hyperparameter_optimization -> model_types_to_tune: ["lstm"]
    # If ModelTrainer always trains all default models, then check for them.
    # For now, let's be flexible.
    model_dirs_exist = any((Path(mt_conf.root_dir) / model_type).exists() for model_type in ["lstm", "bert", "roberta"])
    assert model_dirs_exist, "No model directory created in model_trainer artifacts."
    print("Model Training successful (at least one model attempted).")

    # --- (Optional) Stage 5: Model Evaluation ---
    # print("\n--- Running Model Evaluation ---")
    # evaluation_pipeline = ModelEvaluationPipeline()
    # evaluation_pipeline.main()
    # me_conf = mgr_for_paths.get_model_evaluation_config()
    # assert (Path(me_conf.metric_file_name)).exists()
    # print("Model Evaluation successful.")

    print("\nIntegration test flow completed.")