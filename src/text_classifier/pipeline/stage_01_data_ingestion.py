from text_classifier.config.configuration import ConfigurationManager
from text_classifier.components.data_ingestion import DataIngestion
from text_classifier import logger # Use global logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        try:
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            
            data_ingester = DataIngestion(config=data_ingestion_config)
            data_ingester.download_file() # Step 1: Download
            data_ingester.extract_zip_file() # Step 2: Extract
            saved_csv_path = data_ingester.load_and_validate_data() # Step 3: Load, basic validate, save main CSV
            
            logger.info(f"Main dataset processed and saved at: {saved_csv_path}")
            logger.info(f">>>>>> Stage: {STAGE_NAME} completed successfully <<<<<<\n\nx==========x")
        except Exception as e:
            logger.error(f"Error in {STAGE_NAME}: {e}", exc_info=True)
            logger.info(f">>>>>> Stage: {STAGE_NAME} failed <<<<<<\n\nx==========x")
            raise e

if __name__ == '__main__':
    pipeline = DataIngestionPipeline()
    pipeline.main()

"""
# /home/mohamed-stifi/Desktop/pfa-s4/src/text_classifier/pipeline/stage_01_data_ingestion.py
from text_classifier.config.configuration import ConfigurationManager
from text_classifier.components.data_ingestion import DataIngestion
from text_classifier import logger # Use global logger
from pathlib import Path # Add Path

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        try:
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            
            data_ingester = DataIngestion(config=data_ingestion_config)
            data_ingester.download_file() # Step 1: Download
            data_ingester.extract_zip_file() # Step 2: Extract
            
            # Step 3: Load, validate with Pandera, and get the validated DataFrame
            validated_df = data_ingester.load_and_validate_data() 
            
            # Determine the path where the main CSV (expected by data_validation) should be.
            # This path should match one of the `required_files` in `data_validation_config`.
            # From sample_config.yaml, it's `artifacts/data_ingestion/extracted_data/dummy_data.csv`
            # Or it could be a new file like `main_validated_data.csv` in `data_ingestion_config.root_dir`.
            
            # For simplicity, let's assume `load_and_validate_data` reads the CSV specified by
            # `data_ingestion_config.main_csv_file_name` from `unzip_dir`, validates it,
            # and we just confirm its existence for the next stage.
            # The current `load_and_validate_data` returns the df, it doesn't re-save it.
            # If the `data_validation` stage expects a specific file, we should ensure it's there.
            # The easiest is if `data_ingestion_config.main_csv_file_name` is the target.

            # Let's assume the file checked by data_validation is the one read by load_and_validate_data
            expected_csv_path_str = (Path(data_ingestion_config.unzip_dir) / 
                                     getattr(data_ingestion_config, 'main_csv_file_name', 'AI_Human.csv'))
            
            if not expected_csv_path_str.exists():
                 logger.warning(f"The primary CSV file {expected_csv_path_str} was expected but not found after extraction/validation. "
                                f"The validation stage might fail. Ensure 'main_csv_file_name' in config.yaml is correct.")
            else:
                logger.info(f"Main dataset '{expected_csv_path_str.name}' processed (read and validated). "
                            f"Content shape: {validated_df.shape}")

            logger.info(f">>>>>> Stage: {STAGE_NAME} completed successfully <<<<<<\n\nx==========x")
        except Exception as e:
            logger.error(f"Error in {STAGE_NAME}: {e}", exc_info=True)
            logger.info(f">>>>>> Stage: {STAGE_NAME} failed <<<<<<\n\nx==========x")
            raise e

if __name__ == '__main__':
    pipeline = DataIngestionPipeline()
    pipeline.main()
"""