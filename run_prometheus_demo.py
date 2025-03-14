from datetime import datetime
from prometheus.prometheus import Prometheus
from prometheus.utils import llm_client_openai
from openai import OpenAI
from prometheus.tools.definitions import logging_codes
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

    # initailize the LLM agent with the llm clients.
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

        # provide the path to the tools directory
        tools_path="./tools",
        logger=logger,

        # you can provide tools directly as well, see definitions.py for the 
        # format.
        # extra_tools= [ LLMTool... ]

    )

    # Tasks can then be given to the agent with the following. This will return 
    # the summary that is logged.
    prometheus.Task(input("Task Prompt: "))

    # You are responsable for clearing the history after a task run.
    # prometheus.clear_history()

    # If you want to control the execution of steps rather than letting the 
    # agent control it, you can directly make a plan and call execute step.

    # prometheus.make_plan("plan")
    # prometheus.execute_step()

