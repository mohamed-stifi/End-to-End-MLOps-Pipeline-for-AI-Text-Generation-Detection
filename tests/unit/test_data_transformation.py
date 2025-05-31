import pytest
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
from text_classifier.components.data_transformation import DataTransformation
from text_classifier.entity.config_entity import DataTransformationConfig
from transformers import AutoTokenizer # For type hinting and potential real use if not mocked

# Minimal config for DataTransformation
@pytest.fixture
def dt_config(tmp_path):
    return DataTransformationConfig(
        root_dir=tmp_path / "data_transformation",
        data_path=tmp_path / "raw_data.csv", # Dummy path
        tokenizer_name="prajjwal1/bert-tiny", # A small, fast tokenizer
        max_length=32,
        batch_size=2,
        test_size=0.2,
        val_size=0.1
    )

@pytest.fixture
def data_transformer(dt_config, mocker):
    # Mock nltk.download to prevent actual downloads during tests
    mocker.patch('nltk.download')
    # Mock AutoTokenizer.from_pretrained to return a dummy tokenizer
    mock_tokenizer_instance = MagicMock(spec=AutoTokenizer)
    mock_tokenizer_instance.save_pretrained = MagicMock()
    # Simulate tokenizer output for tokenization tests
    def mock_tokenize(texts, truncation, padding, max_length, return_tensors):
        # A very simplified mock tokenizer output
        num_texts = len(texts)
        return {
            'input_ids': torch.randint(0, 100, (num_texts, max_length)),
            'attention_mask': torch.ones((num_texts, max_length), dtype=torch.long)
        }
    mock_tokenizer_instance.side_effect = mock_tokenize # If __call__ is used
    mock_tokenizer_instance.__call__ = mock_tokenize    # If called directly

    mocker.patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer_instance)
    
    transformer = DataTransformation(config=dt_config)
    # Ensure the mocked tokenizer is set
    transformer.tokenizer = mock_tokenizer_instance
    return transformer

def test_remove_punc(data_transformer):
    assert data_transformer.remove_punc("Hello, world!") == "Hello world"
    assert data_transformer.remove_punc("No punctuation here.") == "No punctuation here"
    assert data_transformer.remove_punc("") == ""
    assert data_transformer.remove_punc(None) == "" # Handled by pd.isna
    assert data_transformer.remove_punc("123!@#") == "123"

def test_is_similar_to_stopword(data_transformer):
    stop_words = {"the", "is", "a"}
    assert data_transformer.is_similar_to_stopword("the", stop_words)
    assert data_transformer.is_similar_to_stopword("The", stop_words) # Case insensitivity
    assert data_transformer.is_similar_to_stopword("th", stop_words) is False # Too short
    assert data_transformer.is_similar_to_stopword("thhe", stop_words) # Fuzzy match
    assert not data_transformer.is_similar_to_stopword("apple", stop_words)

def test_remove_stopwords_fuzzy_optimized(data_transformer):
    text = "This is a test sentence with the word the."
    # data_transformer.stop_words is already initialized (mocked nltk.download)
    # We can override it for more predictable testing if needed
    data_transformer.stop_words = {"is", "a", "the", "with"}
    expected = "This test sentence word" # Simplified expectation
    # The actual fuzzy logic might remove more or less, this needs careful check against impl.
    # For a more robust test, fix the stopwords set
    processed = data_transformer.remove_stopwords_fuzzy_optimized(text, data_transformer.stop_words)
    # This is a bit tricky because fuzzy can be unpredictable.
    # Let's test specific cases based on the `is_similar_to_stopword` logic.
    assert "is" not in processed.split()
    assert "the" not in processed.split()
    assert "This" in processed.split() # Assuming "This" is not a stopword

def test_preprocess_text(data_transformer):
    sample_df = pd.DataFrame({
        'text': ["Hello, world! This is a test.", "Another sentence with Punctuation!!! and stopwords like the"],
        'generated': [0, 1]
    })
    data_transformer.stop_words = {"is", "a", "the", "with", "and", "like"} # Override for predictability

    processed_df = data_transformer.preprocess_text(sample_df.copy()) # Use .copy()
    
    assert "hello world this test" in processed_df['text'].iloc[0] # Check parts
    assert "another sentence punctuation stopwords" in processed_df['text'].iloc[1]
    assert all(p not in txt for txt in processed_df['text'] for p in "!.,")
    assert all(sw not in txt for txt in processed_df['text'] for sw in data_transformer.stop_words)

def test_handle_imbalanced_data(data_transformer):
    # Human texts: 3, AI texts: 1. Should undersample human.
    imbalanced_df = pd.DataFrame({
        'text': ["h1", "h2", "h3", "a1"],
        'generated': [0, 0, 0, 1]
    })
    balanced_df = data_transformer.handle_imbalanced_data(imbalanced_df)
    assert len(balanced_df) == 2
    assert balanced_df['generated'].value_counts()[0] == 1
    assert balanced_df['generated'].value_counts()[1] == 1

    # AI texts: 3, Human texts: 1. Should undersample AI.
    imbalanced_df_2 = pd.DataFrame({
        'text': ["h1", "a1", "a2", "a3"],
        'generated': [0, 1, 1, 1]
    })
    balanced_df_2 = data_transformer.handle_imbalanced_data(imbalanced_df_2)
    assert len(balanced_df_2) == 2
    assert balanced_df_2['generated'].value_counts()[0] == 1
    assert balanced_df_2['generated'].value_counts()[1] == 1

def test_split_dataset(data_transformer, dt_config):
    # Create a dummy balanced dataset
    n_samples = 100
    df = pd.DataFrame({
        'text': [f"text_{i}" for i in range(n_samples)],
        'generated': [i % 2 for i in range(n_samples)] # Perfectly balanced
    })
    dt_config.test_size = 0.2
    dt_config.val_size = 0.1 # of original, so 0.1 / (1-0.2) = 0.125 of train_val

    train_df, val_df, test_df = data_transformer.split_dataset(df)

    expected_test_len = int(n_samples * dt_config.test_size)
    expected_val_len = int((n_samples - expected_test_len) * (dt_config.val_size / (1 - dt_config.test_size)))
    expected_train_len = n_samples - expected_test_len - expected_val_len
    
    assert len(test_df) == expected_test_len
    assert len(val_df) == expected_val_len
    assert len(train_df) == expected_train_len
    # Check for stratification (approximate due to small sample sizes in real splits)
    assert abs(train_df['generated'].mean() - 0.5) < 0.1
    assert abs(val_df['generated'].mean() - 0.5) < 0.1
    assert abs(test_df['generated'].mean() - 0.5) < 0.1


def test_tokenize_data(data_transformer, dt_config):
    texts = pd.Series(["Sample text 1.", "Another one."])
    labels = pd.Series([0, 1])
    
    encodings = data_transformer.tokenize_data(texts, labels)
    
    assert 'input_ids' in encodings
    assert 'attention_mask' in encodings
    assert 'labels' in encodings
    assert encodings['input_ids'].shape == (2, dt_config.max_length)
    assert encodings['labels'].shape == (2,)
    assert data_transformer.tokenizer.call_count > 0 # Check if mocked tokenizer was called

@patch('torch.save')
@patch('pandas.DataFrame.to_csv')
def test_save_data(mock_to_csv, mock_torch_save, data_transformer, tmp_path):
    # Dummy encodings and dataframes
    dummy_encodings = {'input_ids': torch.tensor([[1,2],[3,4]]), 'labels': torch.tensor([0,1])}
    dummy_df = pd.DataFrame({'text': ['a', 'b'], 'generated': [0,1]})
    
    # Ensure root_dir for dt_config is set to tmp_path for this test
    data_transformer.config.root_dir = tmp_path

    data_transformer.save_data(dummy_encodings, dummy_encodings, dummy_encodings,
                               dummy_df, dummy_df, dummy_df)
    
    assert mock_torch_save.call_count == 3 # train, val, test encodings
    mock_torch_save.assert_any_call(dummy_encodings, tmp_path / "train_encodings.pt")
    
    assert mock_to_csv.call_count == 3 # train, val, test dfs
    mock_to_csv.assert_any_call(tmp_path / "train.csv", index=False)
    
    # Check tokenizer save
    data_transformer.tokenizer.save_pretrained.assert_called_once_with(tmp_path / "tokenizer")