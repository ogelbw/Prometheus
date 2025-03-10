from datetime import datetime
from pydantic import BaseModel, Field
from prometheus.prometheus import Prometheus
from prometheus.tools.definitions import System_msg, LLMTool, logging_codes
from prometheus.utils import llm_client_openai, llm_client_interactions
from openai import OpenAI
import logging
from os import getenv as env
from dotenv import load_dotenv

LOGGING_CODE_NAMES = {code.value: code.name for code in logging_codes}

class CustomFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.now().strftime("%H:%M")
        action = LOGGING_CODE_NAMES.get(record.levelno, f"UNKNOWN-{record.levelno}")
        location = record.filename
        msg = record.getMessage()
        return f"[{timestamp}][{action}][{location}] : {msg}"

if __name__ == "__main__":
    load_dotenv()

    # logging
    logging.getLogger("httpx").disabled = True
    logger = logging.getLogger("prometheus")
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)

    # initailize the LLM agent
    prometheus = Prometheus(
        llm_client_executer=llm_client_openai(
            openai=OpenAI(
                api_key=env("OPENAI_API_KEY"),
            ),
            retry_attempts=5,
            model="gpt-4o-mini-2024-07-18"
        ),

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

    # Tasks can then be given to the agent by:
    prometheus.Task(input("Task Prompt: "))

    # class LLMToolParameter(BaseModel):
    #     test: str = Field(..., description="A test parameter.")

    # llm_response = prometheus._llmDevClient.base_invoke(
    #     messages=[
    #         System_msg("List all the tools available to you."),
    #     ],
    #     stream=False,
    #     use_tools=True,
    #     tools={
    #         "A test tool": LLMTool(
    #             name="test_tool",
    #             description="A test tool for testing.",
    #             parameters=LLMToolParameter,
    #             requiredParameters=[],
    #             type="function",
    #             function=test_tool_creation
    #         )
    #     },
    # )

    # print(llm_response.choices[0].message.content)
    # print(llm_response.choices[0].finish_reason)

    # prometheus.CreatePythonTool(
    #     tool_name="list_desktop_files",
    #     tool_description="List all files on the desktop for the current user.",
    #     tool_parameters=[
    #     ],
    #     tool_required_parameters=[
    #     ],
    #     developer_comment="This tool should be able to list all files on the desktop for the current user."
    # )