import pytest
from pathlib import Path
from box import ConfigBox
from text_classifier.utils.common import read_yaml, create_directories # Assuming your utils

# Create a dummy yaml file for testing
@pytest.fixture
def dummy_yaml_file(tmp_path):
    content = {"key": "value", "nested": {"key2": 123}}
    file_path = tmp_path / "dummy.yaml"
    with open(file_path, "w") as f:
        import yaml
        yaml.dump(content, f)
    return file_path

def test_read_yaml(dummy_yaml_file):
    config = read_yaml(dummy_yaml_file)
    assert isinstance(config, ConfigBox)
    assert config.key == "value"
    assert config.nested.key2 == 123

def test_create_directories(tmp_path):
    dir_to_create = tmp_path / "new_dir" / "subdir"
    create_directories([str(dir_to_create)]) # common.py expects list of strings or Path objects
    assert dir_to_create.exists()
    assert dir_to_create.is_dir()

# Add more tests for other utilities and components