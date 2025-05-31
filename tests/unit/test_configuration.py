import pytest
from pathlib import Path
from text_classifier.config.configuration import ConfigurationManager
from text_classifier.entity.config_entity import (
    DataIngestionConfig, DataValidationConfig, DataTransformationConfig,
    ModelTrainerConfig, ModelEvaluationConfig, HyperparameterOptimizationConfig
)

# Define paths to sample config files (assuming they are in tests/fixtures)
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SAMPLE_CONFIG_FILE = FIXTURES_DIR / "sample_config.yaml"
SAMPLE_PARAMS_FILE = FIXTURES_DIR / "sample_params.yaml"

@pytest.fixture
def config_manager(tmp_path):
    # Create dummy artifact root defined in sample_config.yaml inside tmp_path
    # to avoid polluting the actual project structure
    (tmp_path / "test_artifacts").mkdir(exist_ok=True)
    
    # To ensure ConfigurationManager works correctly, we'll use its own logic
    # to create directories, but it needs artifacts_root from the config.
    # We can also patch 'create_directories' if we want to isolate it further.
    # For this test, we'll let it create dirs within tmp_path.
    
    # We also need to ensure the config points to `tmp_path` for its artifacts_root
    # This is tricky as ConfigurationManager reads files directly.
    # Solution: copy sample configs to tmp_path and modify artifacts_root there.

    temp_config_file = tmp_path / "sample_config.yaml"
    temp_params_file = tmp_path / "sample_params.yaml"

    # Read original sample config
    import yaml
    with open(SAMPLE_CONFIG_FILE, 'r') as f:
        config_content = yaml.safe_load(f)
    
    # Override artifacts_root to point to tmp_path
    original_artifacts_root = config_content['artifacts_root']
    new_artifacts_root_name = "test_artifacts_via_fixture"
    config_content['artifacts_root'] = str(tmp_path / new_artifacts_root_name)

    # Adjust all paths within the config to be relative to the new artifacts_root
    def update_paths(data, old_root, new_root_base):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and value.startswith(old_root):
                    data[key] = value.replace(old_root, str(new_root_base/new_artifacts_root_name), 1)
                else:
                    update_paths(value, old_root, new_root_base)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str) and item.startswith(old_root):
                    data[i] = item.replace(old_root, str(new_root_base/new_artifacts_root_name), 1)
                else:
                    update_paths(item, old_root, new_root_base)
    
    update_paths(config_content, original_artifacts_root, tmp_path)


    with open(temp_config_file, 'w') as f:
        yaml.dump(config_content, f)
    
    # Copy params file as is
    with open(SAMPLE_PARAMS_FILE, 'r') as f_params_orig, open(temp_params_file, 'w') as f_params_temp:
        f_params_temp.write(f_params_orig.read())

    manager = ConfigurationManager(config_filepath=temp_config_file, params_filepath=temp_params_file)
    return manager, tmp_path / new_artifacts_root_name


def test_configuration_manager_initialization(config_manager):
    manager, artifacts_root_path = config_manager
    assert manager.config is not None
    assert manager.params is not None
    assert Path(manager.config.artifacts_root).exists()
    assert Path(manager.config.artifacts_root) == artifacts_root_path


def test_get_data_ingestion_config(config_manager):
    manager, artifacts_root_path = config_manager
    config = manager.get_data_ingestion_config()
    assert isinstance(config, DataIngestionConfig)
    # Ensure config.root_dir is Path, or cast artifacts_root_path to str for comparison
    assert Path(config.root_dir) == artifacts_root_path / "data_ingestion" # Compare Path objects
    assert Path(config.root_dir).exists()
    assert config.source_url == "https://example.com/dummy_data.zip"

def test_get_data_validation_config(config_manager):
    manager, artifacts_root_path = config_manager
    config = manager.get_data_validation_config()
    assert isinstance(config, DataValidationConfig)
    assert Path(config.root_dir) == artifacts_root_path / "data_validation" # Compare Path objects
    assert Path(config.root_dir).exists()
    assert Path(config.status_file) == artifacts_root_path / "data_validation/status.txt" # Compare Path

def test_get_data_transformation_config(config_manager):
    manager, artifacts_root_path = config_manager
    config = manager.get_data_transformation_config()
    assert isinstance(config, DataTransformationConfig)
    assert Path(config.root_dir) == artifacts_root_path / "data_transformation" # Compare Path
    assert Path(config.root_dir).exists()
    assert config.tokenizer_name == "prajjwal1/bert-tiny"
    # Ensure data_path is also Path
    assert Path(config.data_path) == artifacts_root_path / "data_ingestion/extracted_data/dummy_data.csv"


def test_get_model_trainer_config(config_manager): # Will address KeyError next
    manager, artifacts_root_path = config_manager
    config = manager.get_model_trainer_config()
    assert isinstance(config, ModelTrainerConfig)
    assert Path(config.root_dir) == artifacts_root_path / "model_trainer" # Compare Path
    assert Path(config.root_dir).exists()
    assert config.num_train_epochs == 1
    assert Path(config.data_path) == artifacts_root_path / "data_transformation"

def test_get_model_evaluation_config(config_manager):
    manager, artifacts_root_path = config_manager
    config = manager.get_model_evaluation_config()
    assert isinstance(config, ModelEvaluationConfig)
    assert Path(config.root_dir) == artifacts_root_path / "model_evaluation" # Compare Path
    assert Path(config.root_dir).exists()
    assert Path(config.metric_file_name) == artifacts_root_path / "model_evaluation/evaluation_report.json" # Compare Path
    assert Path(config.data_path) == artifacts_root_path / "data_transformation"
    assert Path(config.comparison_file) == artifacts_root_path / "model_trainer/model_comparison.json"

def test_get_hyperparameter_optimization_config(config_manager): # Add this missing test case
    manager, artifacts_root_path = config_manager
    config = manager.get_hyperparameter_optimization_config()
    assert isinstance(config, HyperparameterOptimizationConfig)
    assert Path(config.root_dir) == artifacts_root_path / "hyperparameter_optimization" # Compare Path
    assert Path(config.root_dir).exists()
    assert config.n_trials == 1
    assert Path(config.data_path) == artifacts_root_path / "data_transformation"