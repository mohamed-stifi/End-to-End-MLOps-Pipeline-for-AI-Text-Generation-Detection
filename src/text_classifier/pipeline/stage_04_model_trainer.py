from text_classifier.config.configuration import ConfigurationManager
from text_classifier.components.model_trainer import ModelTrainer
from text_classifier import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        try:
            config_manager = ConfigurationManager()
            model_trainer_config = config_manager.get_model_trainer_config()
            # Pass global MLflow URI from main config
            mlflow_tracking_uri = config_manager.config.mlflow_tracking_uri
            
            model_trainer = ModelTrainer(config=model_trainer_config, mlflow_tracking_uri=mlflow_tracking_uri)
            # This trains all models specified in MODEL_TYPES and saves a comparison
            training_summary = model_trainer.train_all_models_and_compare() 
            
            logger.info(f"Model training and comparison complete. Summary: {training_summary}")
            logger.info(f">>>>>> Stage: {STAGE_NAME} completed successfully <<<<<<\n\nx==========x")
        except Exception as e:
            logger.error(f"Error in {STAGE_NAME}: {e}", exc_info=True)
            logger.info(f">>>>>> Stage: {STAGE_NAME} failed <<<<<<\n\nx==========x")
            raise e

if __name__ == '__main__':
    pipeline = ModelTrainingPipeline()
    pipeline.main()