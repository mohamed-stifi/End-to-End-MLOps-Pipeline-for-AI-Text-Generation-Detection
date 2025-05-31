from text_classifier.config.configuration import ConfigurationManager
from text_classifier.components.hyperparameter_optimizer import HyperparameterOptimizer
from text_classifier import logger

STAGE_NAME = "Hyperparameter Optimization Stage"

class HyperparameterOptimizationPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        try:
            config_manager = ConfigurationManager()
            hpo_config = config_manager.get_hyperparameter_optimization_config()
            
            optimizer = HyperparameterOptimizer(config=hpo_config)
            best_params_summary = optimizer.optimize()
            
            logger.info(f"Hyperparameter optimization completed. Summary of best params per model type:")
            for model_type, results in best_params_summary.items():
                logger.info(f"  Model: {model_type}")
                logger.info(f"    Best Value ({hpo_config.metric_to_optimize}): {results['best_value']}")
                logger.info(f"    Best Params: {results['best_params']}")

            logger.info(f"Full HPO summary saved to: {hpo_config.root_dir / 'hpo_optimization_summary.json'}")
            logger.info(f">>>>>> Stage: {STAGE_NAME} completed successfully <<<<<<\n\nx==========x")
        except Exception as e:
            logger.error(f"Error in {STAGE_NAME}: {e}", exc_info=True)
            logger.info(f">>>>>> Stage: {STAGE_NAME} failed <<<<<<\n\nx==========x")
            raise e

if __name__ == '__main__':
    pipeline = HyperparameterOptimizationPipeline()
    pipeline.main()