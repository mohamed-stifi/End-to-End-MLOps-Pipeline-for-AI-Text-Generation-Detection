# /home/mohamed-stifi/Desktop/pfa-s4/tests/unit/test_models.py
import pytest
import torch
from unittest.mock import patch, MagicMock
from text_classifier.models import LSTMClassifier, BERTClassifier, RoBERTaClassifier

BATCH_SIZE = 2
SEQ_LEN = 10
VOCAB_SIZE = 100
EMBEDDING_DIM = 32
HIDDEN_DIM = 64

@pytest.fixture
def dummy_lstm_input():
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    # attention_mask for LSTM often means sequence lengths, or just all ones if handled by pack_padded_sequence
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    # Create some shorter sequences to test padding/packing
    attention_mask[0, SEQ_LEN//2:] = 0 # First sequence is half length
    lengths = attention_mask.sum(dim=1)
    # input_ids should also reflect padding if used by model directly (usually padding_idx=0)
    for i in range(BATCH_SIZE):
        input_ids[i, lengths[i]:] = 0 # Set padding tokens to 0 (assuming 0 is padding_idx)
    
    labels = torch.randint(0, 2, (BATCH_SIZE,))
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


@pytest.fixture
def dummy_transformer_input():
    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN)) # Vocab size for transformers is larger
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    labels = torch.randint(0, 2, (BATCH_SIZE,))
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def test_lstm_classifier_instantiation_and_forward(dummy_lstm_input):
    model = LSTMClassifier(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1 # Simpler for testing
    )
    assert model is not None
    
    logits = model(dummy_lstm_input['input_ids'], dummy_lstm_input['attention_mask'])
    assert logits.shape == (BATCH_SIZE, 2) # num_classes = 2 by default

    # Test training_step, validation_step, test_step
    loss = model.training_step(dummy_lstm_input, 0)
    assert isinstance(loss, torch.Tensor)
    
    # For val/test steps, they append to lists.
    # We need to call on_..._epoch_end to clear these lists and compute metrics.
    model.validation_step(dummy_lstm_input, 0)
    val_metrics = model.on_validation_epoch_end() # This is how PL calls it
    # on_validation_epoch_end logs, doesn't return metrics directly in this impl.
    # We can check if logs were called or directly check the logged values if model.log was mocked.
    # For simplicity, just ensure it runs. If you want to check logged values:
    # with patch.object(model, 'log') as mock_log:
    #    model.on_validation_epoch_end()
    #    mock_log.assert_any_call('val_acc', ANY)

    model.test_step(dummy_lstm_input, 0)
    test_output_dict = model.on_test_epoch_end()
    assert 'test_acc' in test_output_dict

@patch('transformers.AutoModel.from_pretrained')
@patch('transformers.AutoConfig.from_pretrained')
def test_bert_classifier_instantiation_and_forward(mock_config, mock_model, dummy_transformer_input):
    # Mock the transformer model and config loading
    mock_config_instance = MagicMock()
    mock_config_instance.hidden_size = 128 # e.g. bert-tiny
    mock_config.return_value = mock_config_instance
    
    mock_model_instance = MagicMock()
    # BERT output: last_hidden_state, pooler_output
    # For BERTClassifier, we use pooler_output
    mock_bert_output = MagicMock()
    mock_bert_output.pooler_output = torch.randn(BATCH_SIZE, mock_config_instance.hidden_size)
    mock_model_instance.return_value = mock_bert_output # When model() is called
    mock_model.return_value = mock_model_instance # When AutoModel.from_pretrained()

    model = BERTClassifier(model_name='prajjwal1/bert-tiny', num_classes=2, learning_rate=1e-5) # Use a tiny model
    assert model is not None
    
    logits = model(dummy_transformer_input['input_ids'], dummy_transformer_input['attention_mask'])
    assert logits.shape == (BATCH_SIZE, 2)

    loss = model.training_step(dummy_transformer_input, 0)
    assert isinstance(loss, torch.Tensor)
    
    model.validation_step(dummy_transformer_input, 0)
    model.on_validation_epoch_end() # Ensure it runs

    model.test_step(dummy_transformer_input, 0)
    test_output_dict = model.on_test_epoch_end()
    assert 'test_acc' in test_output_dict


@patch('transformers.AutoModel.from_pretrained')
@patch('transformers.AutoConfig.from_pretrained')
def test_roberta_classifier_instantiation_and_forward(mock_config, mock_model, dummy_transformer_input):
    # Mock the transformer model and config loading
    mock_config_instance = MagicMock()
    mock_config_instance.hidden_size = 128
    mock_config.return_value = mock_config_instance
    
    mock_model_instance = MagicMock()
    # RoBERTa output: last_hidden_state, (pooler_output is optional/different)
    # RoBERTaClassifier uses last_hidden_state[:, 0, :]
    mock_roberta_output = MagicMock()
    mock_roberta_output.last_hidden_state = torch.randn(BATCH_SIZE, SEQ_LEN, mock_config_instance.hidden_size)
    mock_model_instance.return_value = mock_roberta_output
    mock_model.return_value = mock_model_instance

    model = RoBERTaClassifier(model_name='roberta-small', num_classes=2, learning_rate=1e-5) # Hypothetical small model
    assert model is not None
    
    logits = model(dummy_transformer_input['input_ids'], dummy_transformer_input['attention_mask'])
    assert logits.shape == (BATCH_SIZE, 2)

    loss = model.training_step(dummy_transformer_input, 0)
    assert isinstance(loss, torch.Tensor)

    model.validation_step(dummy_transformer_input, 0)
    model.on_validation_epoch_end() # Ensure it runs

    model.test_step(dummy_transformer_input, 0)
    test_output_dict = model.on_test_epoch_end()
    assert 'test_acc' in test_output_dict