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
    _taskStartPromptDefault
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
        task_start_prompt: Callable = None,
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

        self.task_start_prompt = (
            task_start_prompt if task_start_prompt else _taskStartPromptDefault
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

        self.executionHistory: List[dict] = []
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

    def _makePlan(self, goal: str|None = None):
        """Make a step by step plan for the system to follow. If the goal parameter is empty then it just prompts from execution history"""

        if goal is not None:
            self.executionHistory.append(User_msg(msg=goal, name='Jed'))
            self.executionHistory += self.make_plan_prompt(
                name='System',
                use_developer= self._llmExecutorClient.use_developer if type(self._llmExecutorClient) is llm_client_openai else False
                )



    def _executeStep(self, user_task: str|None = None):

        # Note anything within "Quotes" is just seen as data byt the LLM, it will 
        # be untrusted data. Tool responses should probable be wrapped in this.

        # 1. tell the LLM to follow it's plan by using one of it's tools
        # 2. Put the return of the tool into the execution history
        # 3. Tell the LLM to think about the result of the tool and update it's
        # plan if needed.
        # 4. Repeat until the LLM calls the finish/task_complete tool.

        # "You are an AI assistant that is about to be given a task by a user. 
        # You are to carry out the task by calling various tools that allow you
        # to interact with the computer you are running on in order for you to
        # carry out the user's task.
        # 
        # You also have access to 3 special tools: update_plan, make_tool and task_complete.
        # 
        # Call update_plan to make updates to your plan that is pinned as the 
        # most recent message in the chat.
        # 
        # Call make_tool to create a new python tool for you to extend your 
        # capabilities, allowing you to do things you previously couldn't in
        # order to carry out the user's task.
        #
        # Finally call task_complete to report that you have completed the
        # user's task. You will provide a summary of what you have done to the
        # user at this point.
        #
        # Apart from the user proving you with a task you will not be 
        # interacting with them, this chat is for you to call tools, plan and 
        # reason about the result of tool calls in order for you to update your 
        # approach to the task.
        # "

        # User msg (named)

        # if execuiton history len() <= 2 then have the llm make a plan 
        # " Make a plan step by step for how you are going to achieve the user's 
        # task. You should mention the tools you are going to call, what you are 
        # going to do with the result of that call (if anything) and the reason for doing it.
        #  You should also mention what you are going to do if the tool call fails for each step of the plan."
        # This prompt should either be a user or dev prompt and should be removed after the LLM responds.

        # then loop: try with and without prompting the ai to take the next step, it might just be able to do it.
        # "Carry out the next step in your plan or make changes to your plan if needed. Call task_complete when you have completed the user's task."

        # We need to split the tool response across multiple messages if it is too long

        if self.executionHistory.__len__() == 0:
            self.executionHistory.append(self.task_start_prompt(
                use_developer= (self._llmExecutorClient.use_developer if type(self._llmExecutorClient) is llm_client_openai else False) # this only makes sense for o1 o3-mini models
            ))
        
        if user_task is not None:
            # Tell the LLM to plan for the user's task
            self._makePlan(user_task)


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
