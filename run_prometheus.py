from prometheus.prometheus import Prometheus
from openai import OpenAI
import logging


if __name__ == "__main__":
    # setting up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # httpx is too verbose
    logging.getLogger("httpx").setLevel(-5)

    prometheus = Prometheus(
        openAI_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="None" # Required by the class but not used by ollama
        ),
        model="qwen2.5:latest",
        tools={},
        tools_path="./tools",
        logger=logger
    )

    prometheus.Task(input("task prompt: "))