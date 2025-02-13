from prometheus.prometheus import Prometheus
from prometheus.utils import llm_client_openai
from openai import OpenAI
import logging
from os import getenv as env
from dotenv import load_dotenv
def test_tool_creation():
    pass

if __name__ == "__main__":
    load_dotenv()

    # setting up logger
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)

    # httpx is too verbose
    logging.getLogger("httpx").disabled = True

    prometheus = Prometheus(
        llm_client_executer = llm_client_openai(
            openai=OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="None" # Required by the class but not used by ollama
            ),
            retry_attempts=5,
            model="qwen2.5:latest",),

        llm_client_dev=llm_client_openai(
            openai=OpenAI(
                api_key=env("OPENAI_API_KEY"),
            ),
            retry_attempts=5,
            model="gpt-4o-mini-2024-07-18"
        ),

        llm_client_reviewer=llm_client_openai(
            openai=OpenAI(
                api_key=env("OPENAI_API_KEY"),
            ),
            retry_attempts=5,
            model="gpt-4o-mini-2024-07-18"
        ),

        tools_path="./tools",
        logger=logger,
    )

    # prometheus.Task(input("task prompt: "))
    prometheus.CreatePythonTool(
        tool_name="list_desktop_files",
        tool_description="List all files on the desktop for the current user.",
        tool_parameters=[
        ],
        tool_required_parameters=[
        ],
        developer_comment="This tool should be able to list all files on the desktop for the current user."
    )