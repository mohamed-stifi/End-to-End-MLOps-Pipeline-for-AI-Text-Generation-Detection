import pytest
import yaml
import json
import os
from pathlib import Path
from box import ConfigBox
from text_classifier.utils.common import (
    read_yaml, create_directories, save_json,
    load_json, save_bin, load_bin, get_size
)

def test_read_yaml_success(tmp_path):
    content = {"key": "value", "nested": {"key2": 123}}
    yaml_file = tmp_path / "test.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(content, f)

    config_box = read_yaml(yaml_file)
    assert isinstance(config_box, ConfigBox)
    assert config_box.key == "value"
    assert config_box.nested.key2 == 123

def test_read_yaml_empty_file(tmp_path):
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.touch()
    with pytest.raises(ValueError, match="yaml file is empty"):
        read_yaml(yaml_file)

def test_read_yaml_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_yaml(Path("non_existent_file.yaml"))

def test_create_directories(tmp_path):
    dir_paths = [tmp_path / "dir1", tmp_path / "dir2/subdir"]
    create_directories(dir_paths, verbose=False)
    assert os.path.exists(dir_paths[0])
    assert os.path.exists(dir_paths[1])

    # Test idempotency (running again should not fail)
    create_directories(dir_paths, verbose=False)
    assert os.path.exists(dir_paths[0])

def test_save_and_load_json(tmp_path):
    data_to_save = {"name": "test_json", "version": 1.0}
    json_file_path = tmp_path / "data.json"

    save_json(json_file_path, data_to_save)
    assert json_file_path.exists()

    loaded_data = load_json(json_file_path)
    assert isinstance(loaded_data, ConfigBox)
    assert loaded_data.name == "test_json"
    assert loaded_data.version == 1.0

def test_save_and_load_bin(tmp_path):
    data_to_save = {"key": [1, 2, 3], "obj": "test_string"}
    bin_file_path = tmp_path / "data.bin"

    save_bin(data_to_save, bin_file_path)
    assert bin_file_path.exists()

    loaded_data = load_bin(bin_file_path)
    assert loaded_data["key"] == [1, 2, 3]
    assert loaded_data["obj"] == "test_string"

def test_get_size(tmp_path):
    file_path = tmp_path / "test_file.txt"
    # Create a file of roughly 1KB and 2KB
    with open(file_path, "wb") as f:
        f.write(os.urandom(1024)) # 1KB
    assert get_size(file_path) == "~ 1 KB"

    with open(file_path, "wb") as f:
        f.write(os.urandom(2048)) # 2KB
    assert get_size(file_path) == "~ 2 KB"