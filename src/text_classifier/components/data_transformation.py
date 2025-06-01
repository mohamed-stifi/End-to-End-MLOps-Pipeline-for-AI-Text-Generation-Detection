import os
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from text_classifier import logger
from text_classifier.entity.config_entity import DataTransformationConfig

# Download required NLTK data
nltk.download('stopwords', quiet=True)

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.df = pd.read_csv(config.data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        self.stop_words = set([w.lower() for w in self.stop_words])

    def remove_punc(self, text):
        """Remove punctuation from text"""
        if pd.isna(text):
            return ""
        new_text = [x for x in str(text) if x not in string.punctuation]
        return ''.join(new_text)

    def is_similar_to_stopword(self, word, stop_words):
        """Check if word is similar to any stopword using fuzzy matching"""
        word = word.lower()
        if len(word) < 3:
            return False
        
        for stop_word in stop_words:
            if abs(len(word) - len(stop_word)) > 3:
                continue
            if fuzz.partial_ratio(word, stop_word) > 80:
                return True
        return False

    def remove_stopwords_fuzzy_optimized(self, text, stop_words):
        """Remove stopwords using fuzzy matching"""
        if pd.isna(text):
            return ""
        
        words = str(text).split()
        filtered_words = [word for word in words if not self.is_similar_to_stopword(word, stop_words)]
        return ' '.join(filtered_words)

    def process_row(self, row, stop_words):
        """Process a single row with fuzzy stopword removal"""
        return self.remove_stopwords_fuzzy_optimized(row, stop_words)

    def process_in_batches(self, df, stop_words, batch_size=1000):
        """Process dataframe in batches for memory efficiency"""
        n_batches = len(df) // batch_size + 1
        results = []
        
        for i in tqdm(range(n_batches), desc="Processing Batches"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(df))
            
            batch_results = Parallel(n_jobs=2)(
                delayed(self.process_row)(row, stop_words) 
                for row in df['text'][batch_start:batch_end]
            )
            results.extend(batch_results)
        
        df_copy = df.copy()
        df_copy['text'] = results
        return df_copy

    def preprocess_text(self, df):
        """Complete text preprocessing pipeline"""
        logger.info("Starting text preprocessing...")
        
        # Remove punctuation
        logger.info("Removing punctuation...")
        df['text'] = df['text'].apply(self.remove_punc)
        
        # Remove stopwords with fuzzy matching
        logger.info("Removing stopwords with fuzzy matching...")
        df = self.process_in_batches(df, self.stop_words)
        
        # Convert to lowercase
        logger.info("Converting to lowercase...")
        df['text'] = df['text'].str.lower()
        
        # Remove extra whitespace
        logger.info("Cleaning whitespace...")
        df['text'] = df['text'].str.strip()
        df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
        
        # Remove empty texts
        df = df[df['text'].str.len() > 0]
        
        logger.info(f"Preprocessing completed. Final dataset shape: {df.shape}")
        return df

    def handle_imbalanced_data(self, df):
        """Handle imbalanced data using undersampling"""
        logger.info("Handling imbalanced data...")
        
        # Separate classes
        human_texts = df[df['generated'] == 0]
        ai_texts = df[df['generated'] == 1]
        
        logger.info(f"Human texts: {len(human_texts)}")
        logger.info(f"AI texts: {len(ai_texts)}")
        
        # Undersample the majority class
        if len(human_texts) > len(ai_texts):
            human_texts_resampled = resample(human_texts, 
                                           replace=False, 
                                           n_samples=len(ai_texts), 
                                           random_state=42)
            balanced_df = pd.concat([human_texts_resampled, ai_texts])
        else:
            ai_texts_resampled = resample(ai_texts, 
                                        replace=False, 
                                        n_samples=len(human_texts), 
                                        random_state=42)
            balanced_df = pd.concat([human_texts, ai_texts_resampled])
        
        logger.info(f"Balanced dataset shape: {balanced_df.shape}")
        logger.info(f"Final class distribution:")
        logger.info(balanced_df['generated'].value_counts())
        
        return balanced_df.reset_index(drop=True)

    def split_dataset(self, df):
        """Split dataset into train, validation, and test sets"""
        logger.info("Splitting dataset...")
        
        # First split: separate test set
        train_val, test = train_test_split(
            df, 
            test_size=self.config.test_size, 
            random_state=42, 
            stratify=df['generated']
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        train, val = train_test_split(
            train_val, 
            test_size=val_size_adjusted, 
            random_state=42, 
            stratify=train_val['generated']
        )
        
        logger.info(f"Train set: {train.shape}")
        logger.info(f"Validation set: {val.shape}")
        logger.info(f"Test set: {test.shape}")
        
        return train, val, test

    def tokenize_data(self, texts, labels=None):
        """Tokenize text data"""
        logger.info("Tokenizing data...")
        
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        if labels is not None:
            encodings['labels'] = torch.tensor(labels.values, dtype=torch.long)
        
        return encodings

    def convert_examples_to_features(self, df_train, df_val, df_test):
        """Convert datasets to features"""
        logger.info("Converting examples to features...")
        
        train_encodings = self.tokenize_data(df_train['text'], df_train['generated'])
        val_encodings = self.tokenize_data(df_val['text'], df_val['generated'])
        test_encodings = self.tokenize_data(df_test['text'], df_test['generated'])
        
        return train_encodings, val_encodings, test_encodings

    def save_data(self, train_encodings, val_encodings, test_encodings, train_df, val_df, test_df):
        """Save processed data"""
        logger.info("Saving processed data...")
        
        # Save encodings
        torch.save(train_encodings, os.path.join(self.config.root_dir, "train_encodings.pt"))
        torch.save(val_encodings, os.path.join(self.config.root_dir, "val_encodings.pt"))
        torch.save(test_encodings, os.path.join(self.config.root_dir, "test_encodings.pt"))
        
        # Save dataframes
        train_df.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(self.config.root_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
        
        logger.info("Data processing completed and saved successfully!")

    def transform_data(self, df=None):
        """Main transformation pipeline"""
        try:
            if df is None: 
                df = self.df
            # Preprocess text
            df_processed = self.preprocess_text(df)
            
            # Handle imbalanced data
            df_balanced = self.handle_imbalanced_data(df_processed)
            
            # Split dataset
            train_df, val_df, test_df = self.split_dataset(df_balanced)
            
            # Convert to features
            train_encodings, val_encodings, test_encodings = self.convert_examples_to_features(
                train_df, val_df, test_df
            )
            
            # Save processed data
            self.save_data(train_encodings, val_encodings, test_encodings, 
                          train_df, val_df, test_df)
            
            return {
                'train_encodings': train_encodings,
                'val_encodings': val_encodings,
                'test_encodings': test_encodings,
                'train_df': train_df,
                'val_df': val_df,
                'test_df': test_df
            }
            
        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise e