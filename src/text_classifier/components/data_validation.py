import os
from text_classifier import logger
from text_classifier.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = True  # Assume all files exist unless proven otherwise
            missing_files = []

            for file_path in self.config.required_files:
                if not os.path.exists(file_path):
                    validation_status = False
                    missing_files.append(file_path)

            # Write status to the status file
            with open(self.config.status_file, 'w') as f:
                if validation_status:
                    f.write("Validation status: True\nAll required files are present.")
                else:
                    f.write(f"Validation status: False\nMissing files: {missing_files}")

            # Optional: log the validation result
            logger.info(f"Data Validation Result: {validation_status}")
            if not validation_status:
                logger.warning(f"Missing files: {missing_files}")

            return validation_status

        except Exception as e:
            logger.error(f"Exception occurred during file validation: {str(e)}")
            raise e
