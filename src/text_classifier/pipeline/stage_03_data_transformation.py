from text_classifier.config.configuration import ConfigurationManager
from text_classifier.components.data_transformation import DataTransformation
from text_classifier import logger

STAGE_NAME = "Data Transformation Stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        try:
            config_manager = ConfigurationManager()
            data_transformation_config = config_manager.get_data_transformation_config()
            
            data_transformer = DataTransformation(config=data_transformation_config)
            data_transformer.transform_data() # This method handles all sub-steps
            
            logger.info(f"Data transformation complete. Processed data saved to: {data_transformation_config.root_dir}")
            logger.info(f">>>>>> Stage: {STAGE_NAME} completed successfully <<<<<<\n\nx==========x")
        except Exception as e:
            logger.error(f"Error in {STAGE_NAME}: {e}", exc_info=True)
            logger.info(f">>>>>> Stage: {STAGE_NAME} failed <<<<<<\n\nx==========x")
            raise e

if __name__ == '__main__':
    pipeline = DataTransformationPipeline()
    pipeline.main()