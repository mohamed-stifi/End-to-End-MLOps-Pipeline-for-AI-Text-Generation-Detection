from text_classifier.config.configuration import ConfigurationManager
from text_classifier.components.data_validation import DataValidation
from text_classifier import logger

STAGE_NAME = "Data Validation Stage"

class DataValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        try:
            config_manager = ConfigurationManager()
            data_validation_config = config_manager.get_data_validation_config()
            
            data_validator = DataValidation(config=data_validation_config)
            is_valid = data_validator.validate_all_files_exist()
            
            if not is_valid:
                logger.error(f"{STAGE_NAME} failed: Not all required files are valid or present.")
                # Depending on severity, you might want to raise an exception to stop the pipeline
                # raise Exception("Data validation failed, stopping pipeline.")
            
            logger.info(f"Data validation status: {'Valid' if is_valid else 'Invalid'}")
            logger.info(f">>>>>> Stage: {STAGE_NAME} completed successfully <<<<<<\n\nx==========x")
        except Exception as e:
            logger.error(f"Error in {STAGE_NAME}: {e}", exc_info=True)
            logger.info(f">>>>>> Stage: {STAGE_NAME} failed <<<<<<\n\nx==========x")
            raise e

if __name__ == '__main__':
    pipeline = DataValidationPipeline()
    pipeline.main()