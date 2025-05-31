import json
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError
from text_classifier.constants import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TASKS_TOPIC # We'll define these constants

logger = logging.getLogger(__name__)

# Ensure KAFKA_BOOTSTRAP_SERVERS is defined in your constants or fetched from env
KAFKA_BOOTSTRAP_SERVERS_ENV = KAFKA_BOOTSTRAP_SERVERS # "localhost:9092" # Example
KAFKA_TASKS_TOPIC_ENV = KAFKA_TASKS_TOPIC # "mlops_tasks" # Example


class TaskProducer:
    def __init__(self, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS_ENV):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                retries=5 # Configure retries for robustness
            )
            logger.info(f"KafkaProducer initialized successfully for servers: {self.bootstrap_servers}")
        except KafkaError as e:
            logger.error(f"Failed to initialize KafkaProducer: {e}", exc_info=True)
            # Depending on your app's needs, you might want to raise this or handle it
            # For now, subsequent calls to send_task will fail gracefully if producer is None

    def send_task(self, topic: str, task_type: str, payload: dict = None):
        if not self.producer:
            logger.error("Kafka producer is not initialized. Cannot send task.")
            return False

        message = {
            "task_type": task_type,
            "payload": payload if payload is not None else {}
        }
        try:
            future = self.producer.send(topic, value=message)
            # Block for 'synchronous' sends, or handle asynchronously
            record_metadata = future.get(timeout=10) # Adjust timeout as needed
            logger.info(f"Task '{task_type}' sent to Kafka topic '{topic}' successfully. Offset: {record_metadata.offset}")
            return True
        except KafkaError as e:
            logger.error(f"Failed to send task '{task_type}' to Kafka topic '{topic}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while sending task '{task_type}': {e}", exc_info=True)
            return False

    def close(self):
        if self.producer:
            self.producer.flush() # Ensure all pending messages are sent
            self.producer.close()
            logger.info("KafkaProducer closed.")

# Global instance (can be managed better with FastAPI lifespan events or dependency injection)
# For simplicity here, a global instance. Be mindful of its lifecycle if used in a complex app.
task_producer = TaskProducer()

# Optional: Function to ensure producer is closed on application shutdown
def close_kafka_producer():
    task_producer.close()