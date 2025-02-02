from prometheus.prometheus import Prometheus
from prometheus.utils import llm_client_openai
from openai import OpenAI
import logging

def test_tool_creation():
    pass

if __name__ == "__main__":
    # setting up logger
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)

    # httpx is too verbose
    logging.getLogger("httpx").disabled = True

    prometheus = Prometheus(
        llm_client = llm_client_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="None" # Required by the class but not used by ollama
            )),
        tools_path="./tools",
        logger=logger,
        dev_llm_model="openchat:7b",
        reviewer_llm_model="openchat:7b"
    )

    prometheus.Task(input("task prompt: "))