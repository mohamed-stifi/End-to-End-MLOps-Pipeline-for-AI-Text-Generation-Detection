from text_classifier import logger
from text_classifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from text_classifier.pipeline.stage_02_data_validation import DataValidationPipeline
from text_classifier.pipeline.stage_03_data_transformation import DataTransformationPipeline
from text_classifier.pipeline.stage_04_model_trainer import ModelTrainingPipeline
from text_classifier.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from text_classifier.pipeline.stage_06_hyperparameter_optimization import HyperparameterOptimizationPipeline 
import argparse

stages = {
    "ingestion": DataIngestionPipeline,
    "validation": DataValidationPipeline,
    "transformation": DataTransformationPipeline,
    "training": ModelTrainingPipeline,         # Original training (could be with default params)
    "hpo": HyperparameterOptimizationPipeline, # Hyperparameter Optimization stage
    "evaluation": ModelEvaluationPipeline,
}

def run_pipeline(stage: str = "all"):
    # Define a typical full pipeline order
    full_pipeline_order = [
        DataIngestionPipeline,
        DataValidationPipeline,
        DataTransformationPipeline,
        HyperparameterOptimizationPipeline, # Run HPO
        ModelTrainingPipeline,              # Then train with (potentially updated) best params
        ModelEvaluationPipeline,
    ]


    if stage == "all":
        logger.info("Starting full MLOps pipeline execution (including HPO)...")
        current_params = None # Placeholder for params, can be loaded/updated
        for pipeline_class in full_pipeline_order:
            try:
                # Stage specific logic here, e.g., HPO output might influence ModelTraining input
                # For now, stages run sequentially.
                # A more advanced setup might pass HPO results to ModelTrainer.
                
                # Example: If HPO runs, its output (best_params.json) could be read by
                # ConfigurationManager before get_model_trainer_config to update params for training.
                # This requires modifying ConfigurationManager to optionally load params from HPO output.

                pipeline_instance = pipeline_class()
                logger.info(f"--- Running stage: {pipeline_class.__name__} ---")
                pipeline_instance.main()
            except Exception as e:
                logger.error(f"Pipeline stage {pipeline_class.__name__} failed: {e}", exc_info=True)
                raise # Stop full pipeline on error in one stage
        logger.info("Full MLOps pipeline executed successfully.")
            
    elif stage in stages:
        logger.info(f"Starting pipeline stage: {stage}...")
        try:
            pipeline_class = stages[stage]()
            pipeline_class.main()
            logger.info(f"Pipeline stage: {stage} executed successfully.")
        except Exception as e:
            logger.error(f"Pipeline stage {stage} failed: {e}", exc_info=True)
            raise e # Re-raise to indicate failure
    else:
        logger.error(f"Invalid stage '{stage}'. Available stages: {list(stages.keys())} or 'all'.")
        raise ValueError(f"Invalid stage: {stage}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MLOps pipeline stages.")
    parser.add_argument(
        "--stage", 
        type=str, 
        default="all", 
        help=f"Specify pipeline stage to run: {list(stages.keys()) + ['all']} (default: all)."
    )
    args = parser.parse_args()
    
    try:
        run_pipeline(args.stage)
    except Exception as e:
        logger.critical(f"Pipeline execution aborted due to an error in stage: {args.stage}. Error: {e}")
        # sys.exit(1) # Optional: exit with error code