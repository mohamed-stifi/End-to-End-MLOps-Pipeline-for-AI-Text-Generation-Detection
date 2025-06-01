from text_classifier import logger
from text_classifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from text_classifier.pipeline.stage_02_data_validation import DataValidationPipeline
from text_classifier.pipeline.stage_03_data_transformation import DataTransformationPipeline
from text_classifier.pipeline.stage_04_model_trainer import ModelTrainingPipeline
from text_classifier.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from text_classifier.pipeline.stage_06_hyperparameter_optimization import HyperparameterOptimizationPipeline
from text_classifier.pipeline.stage_07_update_parameters import UpdateParametersPipeline # New import
import argparse

stages = {
    "ingestion": DataIngestionPipeline,
    "validation": DataValidationPipeline,
    "transformation": DataTransformationPipeline,
    "hpo": HyperparameterOptimizationPipeline,
    "update_params": UpdateParametersPipeline,
    "training": ModelTrainingPipeline,
    "evaluation": ModelEvaluationPipeline,
}

def run_pipeline(stage: str = "all"):
    # Define a typical full pipeline order
    full_pipeline_order = [
        DataIngestionPipeline,
        DataValidationPipeline,
        DataTransformationPipeline,
        HyperparameterOptimizationPipeline, # 1. Run HPO
        UpdateParametersPipeline,           # 2. Update params.yaml with HPO results
        ModelTrainingPipeline,              # 3. Then train with updated best params
        ModelEvaluationPipeline,
    ]

    if stage == "all":
        logger.info("Starting full MLOps pipeline execution (including HPO and param update)...")
        for pipeline_class in full_pipeline_order:
            try:
                pipeline_instance = pipeline_class()
                # For ConfigurationManager to pick up the updated params.yaml in ModelTrainingPipeline,
                # it will naturally re-read it when instantiated within that pipeline's main method.
                logger.info(f"--- Running stage: {pipeline_class.__name__} ---")
                pipeline_instance.main()
            except Exception as e:
                logger.error(f"Pipeline stage {pipeline_class.__name__} failed: {e}", exc_info=True)
                raise
        logger.info("Full MLOps pipeline executed successfully.")
            
    elif stage in stages:
        logger.info(f"Starting pipeline stage: {stage}...")
        try:
            pipeline_instance = stages[stage]()
            pipeline_instance.main()
            logger.info(f"Pipeline stage: {stage} executed successfully.")
        except Exception as e:
            logger.error(f"Pipeline stage {stage} failed: {e}", exc_info=True)
            raise e
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