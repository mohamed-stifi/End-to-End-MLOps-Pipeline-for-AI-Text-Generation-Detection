
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class BERTClassifier(pl.LightningModule):
    def __init__(self, model_name='bert-base-uncased', num_classes=2, learning_rate=2e-5, 
                 dropout=0.1, warmup_steps=0, max_epochs=10):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained BERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)

        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        self.validation_epoch_outputs = []
        self.test_step_outputs = []
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classify
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
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
        attention_mask = batch['attention_mask']
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
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)

        self.test_step_outputs.append(
            {
            'preds': preds,
            'labels': labels
        }
        )
        
        return {
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        if self.hparams.warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.1, 
                total_iters=self.hparams.warmup_steps
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                }
            }
        return optimizer