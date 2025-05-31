import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

class LSTMClassifier(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, 
                 dropout=0.3, learning_rate=1e-3, num_classes=2):
        super().__init__()
        self.save_hyperparameters()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        self.criterion = nn.CrossEntropyLoss()


        self.validation_epoch_outputs = []
        self.test_step_outputs = []
        
    def forward(self, input_ids, attention_mask=None):
        # Get sequence lengths for packing
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
        else:
            lengths = torch.full((input_ids.size(0),), input_ids.size(1))
        
        # Embedding
        embedded = self.embedding(input_ids)
        
        # Pack sequences for efficiency
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, 
                                             enforce_sorted=False)
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Use the last hidden state from both directions
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        # We want the last layer's forward and backward hidden states
        forward_hidden = hidden[-2]  # Last layer, forward direction
        backward_hidden = hidden[-1]  # Last layer, backward direction
        
        # Concatenate forward and backward hidden states
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Apply dropout and classify
        output = self.dropout(final_hidden)
        logits = self.classifier(output)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)

        self.validation_epoch_outputs.append(
            {
            'val_loss': loss,
            'preds': preds,
            'labels': labels
        }
        )
        
        return {
            'val_loss': loss,
            'preds': preds,
            'labels': labels
        }
    
    def on_validation_epoch_end(self):
        outputs = self.validation_epoch_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_labels = torch.cat([x['labels'] for x in outputs])
        
        acc = (all_preds == all_labels).float().mean()
        
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        self.validation_epoch_outputs = []
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)

        self.test_step_outputs.append(
            {
            'test_loss': loss,
            'preds': preds,
            'labels': labels
        }
        )
        
        return {
            'test_loss': loss,
            'preds': preds,
            'labels': labels
        }
    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        all_preds = torch.cat([x['preds'] for x in outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy()
        
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        self.log('test_acc', acc)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1)

        self.test_step_outputs = []
        
        return {
            'test_acc': acc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1
        }
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
