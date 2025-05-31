import json
import logging
import time
import os
import sys
from kafka import KafkaConsumer
from kafka.errors import KafkaError

# Ensure the src directory is in PYTHONPATH if running this script directly
# This allows importing from text_classifier
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from text_classifier import logger # Use the project's configured logger
from text_classifier.constants import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TASKS_TOPIC
from text_classifier.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from text_classifier.pipeline.stage_06_hyperparameter_optimization import HyperparameterOptimizationPipeline
# If you have a main pipeline runner for "full_retrain" or specific stages:
from main import run_pipeline # Assuming your project's main.py has run_pipeline

# Setup basic logging if this script is run standalone and text_classifier logger isn't fully set up
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s: %(levelname)s: %(module)s: %(name)s: %(message)s]',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

class TaskConsumer:
    def __init__(self, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, topic=KAFKA_TASKS_TOPIC, group_id="mlops_task_processor_group"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.consumer = None
        while self.consumer is None:
            try:
                self.consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.bootstrap_servers,
                    auto_offset_reset='earliest', # Process messages from the beginning if consumer is new
                    enable_auto_commit=True,      # Commit offsets automatically after processing
                    group_id=self.group_id,
                    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                    consumer_timeout_ms=10000 # Timeout to allow checking for shutdown
                )
                logger.info(f"KafkaConsumer initialized successfully for topic '{self.topic}' and group '{self.group_id}'.")
            except KafkaError as e:
                logger.error(f"Failed to initialize KafkaConsumer (will retry): {e}", exc_info=True)
                time.sleep(5) # Wait before retrying
            except Exception as e:
                logger.error(f"Unexpected error during KafkaConsumer initialization (will retry): {e}", exc_info=True)
                time.sleep(5)


    def process_message(self, message_data: dict):
        task_type = message_data.get("task_type")
        payload = message_data.get("payload", {})
        logger.info(f"Received task: {task_type} with payload: {payload}")

        try:
            if task_type == "evaluate_models":
                logger.info("Starting model evaluation pipeline...")
                pipeline = ModelEvaluationPipeline()
                pipeline.main()
                logger.info("Model evaluation pipeline completed.")
            elif task_type == "tune_hyperparameters":
                logger.info("Starting hyperparameter optimization pipeline...")
                pipeline = HyperparameterOptimizationPipeline()
                pipeline.main()
                logger.info("Hyperparameter optimization pipeline completed.")
            elif task_type == "full_retrain":
                logger.info("Starting full retraining pipeline...")
                # Assuming run_pipeline("all") triggers the full sequence
                # You might want to define a more specific "retrain" sequence
                # that skips ingestion/validation if data hasn't changed.
                run_pipeline(stage="all") # Or a specific retraining stage sequence
                logger.info("Full retraining pipeline completed.")
            else:
                logger.warning(f"Unknown task type received: {task_type}")
        except Exception as e:
            logger.error(f"Error processing task '{task_type}': {e}", exc_info=True)
            # Optionally, send error status back to another Kafka topic or a dead-letter queue

    def listen(self):
        if not self.consumer:
            logger.error("Kafka consumer is not initialized. Cannot listen for messages.")
            return

        logger.info(f"Listening for messages on Kafka topic '{self.topic}'...")
        try:
            for message in self.consumer:
                logger.info(f"Consumed message: offset={message.offset}, key={message.key}, value={message.value}")
                self.process_message(message.value)
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user. Shutting down...")
        except Exception as e:
            logger.error(f"An error occurred in the Kafka consumer loop: {e}", exc_info=True)
        finally:
            if self.consumer:
                self.consumer.close()
                logger.info("KafkaConsumer closed.")

if __name__ == "__main__":
    # Ensure KAFKA_BOOTSTRAP_SERVERS is correctly set as an environment variable
    # or defined in constants.py and accessible
    consumer_service = TaskConsumer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, # Fetches from constants
        topic=KAFKA_TASKS_TOPIC
    )
    consumer_service.listen()