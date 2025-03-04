from typing import Callable, List, Dict, Any
from os import path, listdir
from logging import Logger, INFO
import importlib.util
from prometheus.tools.definitions import *
from prometheus.utils import *
from prometheus.devTeam.toolCreator import Python_Tool_developer
from prometheus.default_prompts import (
    _makeToolPromptDefault,
    _makePlanPromptDefault,
    _updatePlanPromptDefault,
    _takeActionStepPromptDefault,
)


class Prometheus:
    """
    Self tooling LLM agent.
    """

    def __init__(
        self,
        llm_client_executer: llm_client_base,
        llm_client_dev: llm_client_base,
        llm_client_reviewer: llm_client_base,
        tools_path: str = "./tools",  # Path to the tools directory
        make_tool_prompt: Callable = None,
        make_plan_prompt: Callable = None,
        execution_prompt: Callable = None,
        step_retry_attempts: int = 5,
        logger: Logger = None,
    ) -> None:
        self.logger = logger

        # import all the external tools
        self.tools_path = tools_path
        self.tools = {}
        self._import_tools()

        # setting the default prompts
        self.create_tool_prompt = (
            make_tool_prompt if make_tool_prompt else _makeToolPromptDefault
        )
        self.make_plan_prompt = (
            make_plan_prompt if make_plan_prompt else _makePlanPromptDefault
        )
        self.update_plan_prompt = (
            make_plan_prompt if make_plan_prompt else _updatePlanPromptDefault
        )
        self.take_step_prompt = (
            execution_prompt if execution_prompt else _takeActionStepPromptDefault
        )

        self._llmExecutorClient = llm_client_executer
        self._llmDevClient = llm_client_dev
        self._llmReviewerClient = llm_client_reviewer
        self.llm_interactions = llm_client_interactions(self)
        self.pythonToolMaker = Python_Tool_developer(
            tools_path=tools_path,
            step_retry_attempts=step_retry_attempts,
            logger=self.log,
            llm_dev_client=self._llmDevClient,
            llm_reviewer_client=self._llmReviewerClient,
            make_tool_prompt_template=self.create_tool_prompt,
            review_tool_prompt_template=self.create_tool_prompt,  # TODO <-- Make a review tool prompt
            pip_tool=self.tools["pip"],
        )

        self.currentPlan: str = ""
        """ The queue of instructions to be executed by the system."""

        self.goal = ""
        """ The goal given to the system by the user (the main task)."""

        self.executionHistory: List[InstructionResponse] = []
        """ The chat history of the llm client."""

        self.step_retry_attempts = step_retry_attempts
        """ The number of times to retry a step if it fails."""

    def log(self, message: str, level: int = INFO):
        """Logs a message to the logger."""
        if self.logger:
            self.logger.log(level, message)

    def _import_tool(self, tools_path: str, tool_name: str):
        """Imports a tool from the tools directory as a module."""

        # loading the python file as a module
        spec = importlib.util.spec_from_file_location(
            tool_name, path.join(tools_path, f"{tool_name}.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def _import_tools(self):
        """Imports all of the tools in the tools directory."""
        tools = {}
        tools_path = path.abspath(self.tools_path)
        for tool_name in [
            x.split(".")[0] for x in listdir(tools_path) if x.endswith(".py")
        ]:
            tools[tool_name] = self._import_tool(tools_path, tool_name)

            # add the tool to the tools dictionary using the ToolDescription
            # function in all tool scripts
            self.tools[tool_name] = tools[tool_name].ToolDescription()
            self.tools[tool_name].function = tools[tool_name].Run
            self.log(f"Tool {tool_name} imported successfully.")

    def _callTool(self, tool_name: str, tool_args: Dict[str, Any]):
        """Calls a tool with the given name and arguments."""
        return self.tools[tool_name].function(**tool_args)

    def CreatePythonTool(
        self,
        tool_name: str,
        tool_description: str,
        tool_parameters: List[LLMToolParameter],
        tool_required_parameters: List[str],
        developer_comment: str,
    ):
        """Creates a python tool in the tools directory."""
        self.pythonToolMaker.MakeTool(
            tool_name,
            tool_description,
            tool_parameters,
            tool_required_parameters,
            developer_comment,
        )

        # add the tool to the list of tools
        module = self._import_tool(self.tools_path, tool_name)
        self.tools[tool_name] = module.ToolDescription()
        self.tools[tool_name].function = module.Run
        self.log(f"Tool {tool_name} created successfully.")

    def _makePlan(self, goal: str):
        """Make a step by step plan for the system to follow."""
        plan_prompt = self.executionHistory + self.make_plan_prompt(goal)

        # Get the llm to generate a plan message that will be 'pinned' to the
        # message history. Note use_tools is set to True so that the LLm knows
        # about the tool it have access to, If false the api doesn't send them.
        for i in range(self.step_retry_attempts):
            llm_response = self._llmDevClient.base_invoke(
                messages=plan_prompt,
                stream=False,
                use_tools=True,
                tools=self.llm_interactions._getFormattedTools(self.tools),
            )

            if llm_response.choices[0].finish_reason != "stop":
                self.log(
                    f"Failed to make a plan, llm returned stop reason: {llm_response.choices[0].finish_reason}",
                    level=logging_codes.PLAN_FAILED.value,
                )
                continue
            self.currentPlan = llm_response.choices[0].message.content
            self.log(
                f"Plan made: \n{self.currentPlan}\n",
                level=logging_codes.PLAN_MADE.value,
            )


    def _executeStep(self):
        pass

    def generate_summary(self):
        """Generate a summary of the system's actions."""

        messages = filter_system_messages(self.executionHistory) + [
            {
                "role": "system",
                "content": f"You are an AI responsible for generating a summary of the system's actions.",
            },
            {
                "role": "system",
                "content": f"Generate a summary of the system's actions to who informed the system to: {self.goal}.",
            },
        ]

        return self.llm_interactions._getLLMResponseMessage(
            self._llmExecutorClient.base_invoke(
                messages=messages,
                stream=False,
                use_tools=False,
            )
        )

    def Task(self, goal: str):
        """Perform a task given by the user."""
        self.goal = goal

        # make a plan to achieve the goal
        self._makePlan(goal)

        # execute the plan
        while self.currentPlan.__len__() > 0:
            self._executeStep()
        else:
            self.log("Task complete.\n", level=logging_codes.TASK_COMPLETE.value)
            self.log(self.generate_summary(), level=logging_codes.TASK_COMPLETE.value)
