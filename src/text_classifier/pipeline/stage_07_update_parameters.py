from pathlib import Path
from text_classifier.config.configuration import ConfigurationManager
from text_classifier.utils.common import update_params_from_hpo
from text_classifier import logger
from text_classifier.constants import CONFIG_DIR # To get params.yaml path

STAGE_NAME = "Update Parameters Stage"

class UpdateParametersPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        try:
            config_manager = ConfigurationManager() # To get paths from config
            
            hpo_config = config_manager.get_hyperparameter_optimization_config()
            hpo_summary_file = hpo_config.root_dir / "hpo_optimization_summary.json"
            
            params_yaml_file = CONFIG_DIR / "params.yaml"

            # Map HPO summary keys to params.yaml top-level keys
            # This needs to match your params.yaml structure
            model_config_map = {
                "lstm": "LSTM",
                "bert": "BERT",
                "roberta": "RoBERTa"
            }

            logger.info(f"Attempting to update '{params_yaml_file}' from HPO summary '{hpo_summary_file}'")
            success = update_params_from_hpo(
                hpo_summary_path=Path(hpo_summary_file),
                params_yaml_path=params_yaml_file,
                model_config_map=model_config_map
            )
            
            if success:
                logger.info(f"Parameters in '{params_yaml_file}' updated successfully based on HPO.")
            else:
                logger.warning(f"Parameters update from HPO was not fully successful or no updates were made. Check previous logs.")

            logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")

        except Exception as e:
            logger.error(f"Error in {STAGE_NAME}: {e}", exc_info=True)
            logger.info(f">>>>>> Stage: {STAGE_NAME} failed <<<<<<\n\nx==========x")
            raise e

if __name__ == '__main__':
    pipeline = UpdateParametersPipeline()
    pipeline.main()