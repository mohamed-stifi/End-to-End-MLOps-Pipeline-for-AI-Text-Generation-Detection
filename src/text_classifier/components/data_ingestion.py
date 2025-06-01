import os
import urllib.request as request
import zipfile
import pandas as pd
from text_classifier import logger
from text_classifier.utils.common import get_size
from text_classifier.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_url,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

        csv_files = list(Path(self.config.unzip_dir).glob("*.csv"))
        data_path = csv_files[0]  # Take the first CSV file
        df = pd.read_csv(data_path)

        if self.config.max_sample_number < df.shape[0]:
            df_c1 = df[df['generated'] == 1.0][:self.config.max_sample_number]
            df_c0 = df[df['generated'] == 0.0][:self.config.max_sample_number + 10]

            df_subset = pd.concat([df_c1, df_c0], ignore_index=True)

            df_subset.to_csv(data_path)

    def load_and_validate_data(self):
        """Load and validate the dataset"""
        try:
            # Assuming the CSV file is in the extracted directory
            csv_files = list(Path(self.config.unzip_dir).glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in extracted directory")
            
            data_path = csv_files[0]  # Take the first CSV file
            df = pd.read_csv(data_path)
            
            # Validate columns
            required_columns = ['text', 'generated']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Log data statistics
            logger.info(f"Dataset loaded successfully with shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Human texts: {(df['generated'] == 0).sum()}")
            logger.info(f"AI texts: {(df['generated'] == 1).sum()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e
        
        





'''          
import urllib.request as request
import zipfile
import pandas as pd
from text_classifier import logger
from text_classifier.utils.common import get_size
from text_classifier.entity.config_entity import DataIngestionConfig
from pathlib import Path
import pandera as pa # Import Pandera

# Define a Pandera schema for expected data
# Adjust checks as per actual data expectations
DATA_SCHEMA = pa.DataFrameSchema({
    "text": pa.Column(
        str,
        checks=pa.Check(
            lambda s: s.str.len() > 0,
            element_wise=False,  # Apply to the series as a whole
            error="Text column cannot contain empty strings after initial load"
        ),
        nullable=False # Explicitly state text cannot be null if that's the case
    ),
    "generated": pa.Column(
        int,
        checks=pa.Check.isin(
            [0, 1],
            error="Generated column must be 0 or 1"  # <-- Corrected: error is part of the Check
        ),
        nullable=False
    )
    # Add more columns and checks as needed
})

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_url,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")


    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)


    def load_and_validate_data(self) -> pd.DataFrame: # Return DataFrame for further use
        """Load and validate the dataset using Pandera"""
        try:
            # Assuming the CSV file is in the extracted directory
            # Use main_csv_file_name from config if it exists, otherwise glob
            """
            if hasattr(self.config, 'main_csv_file_name') and self.config.main_csv_file_name:
                 data_path = Path(self.config.unzip_dir) / self.config.main_csv_file_name
                 if not data_path.exists():
                     raise FileNotFoundError(f"Specified main CSV file {data_path} not found in extracted directory.")
            else: # Fallback to globbing if main_csv_file_name is not defined or empty
            """
            csv_files = list(Path(self.config.unzip_dir).glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in extracted directory")
            data_path = csv_files[0]  # Take the first CSV file
        
            logger.info(f"Loading data from: {data_path}")
            df = pd.read_csv(data_path)
            
            # Basic column existence check (Pandera will also do this)
            required_columns_initial = ['text', 'generated']
            missing_columns = [col for col in required_columns_initial if col not in df.columns]
            if missing_columns:
                # Log and raise before Pandera validation for clearer error on missing columns
                logger.error(f"Missing required columns for basic loading: {missing_columns}")
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Validate DataFrame using Pandera schema
            logger.info("Validating data schema with Pandera...")
            try:
                validated_df = DATA_SCHEMA.validate(df, lazy=True) # lazy=True collects all errors
                logger.info("Pandera schema validation successful.")
            except pa.errors.SchemaErrors as err:
                logger.error(f"Pandera schema validation failed:\n{err.failure_cases}")
                # You might want to save err.failure_cases to a file or log more details
                # Depending on severity, you might raise an exception to stop the pipeline
                raise ValueError(f"Data schema validation failed. Check logs for details. First error: {err.failure_cases.iloc[0]['failure_case']}") from err
            
            # Log data statistics
            logger.info(f"Dataset loaded and validated successfully with shape: {validated_df.shape}")
            logger.info(f"Columns: {list(validated_df.columns)}")
            if 'generated' in validated_df.columns:
                logger.info(f"Human texts: {(validated_df['generated'] == 0).sum()}")
                logger.info(f"AI texts: {(validated_df['generated'] == 1).sum()}")
            
            # Save the validated main CSV to a known location if needed by subsequent stages
            # or ensure data_validation stage uses this directly.
            # For now, DataValidation stage in your code checks file existence.
            # If DataIngestion is supposed to *produce* the final CSV for validation stage, save it here.
            # Example:
            # main_validated_csv_path = Path(self.config.root_dir) / "main_validated_data.csv"
            # validated_df.to_csv(main_validated_csv_path, index=False)
            # logger.info(f"Validated main dataset saved to: {main_validated_csv_path}")
            # return main_validated_csv_path # Then DataValidation would check this file

            return validated_df # Return the validated DataFrame
            
        except Exception as e:
            logger.error(f"Error loading and validating data: {e}", exc_info=True)
            raise e
'''