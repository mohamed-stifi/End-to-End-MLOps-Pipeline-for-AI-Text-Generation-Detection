from text_classifier.config.configuration import ConfigurationManager
from text_classifier.components.model_evaluation import ModelEvaluation
from text_classifier import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        try:
            config_manager = ConfigurationManager()
            model_evaluation_config = config_manager.get_model_evaluation_config()
            
            model_evaluator = ModelEvaluation(config=model_evaluation_config)
            evaluation_report = model_evaluator.evaluate_all_trained_models()
            
            if evaluation_report:
                logger.info(f"Model evaluation complete. Report generated: {model_evaluation_config.metric_file_name}")
                logger.info(f"Best model from evaluation: {evaluation_report.get('best_model_from_evaluation', {}).get('name')}")
            else:
                logger.warning("Model evaluation did not produce a report.")
            
            logger.info(f">>>>>> Stage: {STAGE_NAME} completed successfully <<<<<<\n\nx==========x")
        except Exception as e:
            logger.error(f"Error in {STAGE_NAME}: {e}", exc_info=True)
            logger.info(f">>>>>> Stage: {STAGE_NAME} failed <<<<<<\n\nx==========x")
            raise e

if __name__ == '__main__':
    pipeline = ModelEvaluationPipeline()
    pipeline.main()