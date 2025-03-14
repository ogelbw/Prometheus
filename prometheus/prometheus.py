from typing import Callable, List, Dict, Any
from os import path, listdir
from logging import Logger, INFO
import importlib.util
from prometheus.tools.definitions import *
from prometheus.utils import *
from prometheus.devTeam.toolCreator import Python_Tool_developer
from prometheus.default_prompts import (
    _makeToolSummerizeHistoryPromptDefault,
    _makePlanPromptDefault,
    _takeActionStepPromptDefault,
    _taskStartPromptDefault,
    _makeToolReviewerStartPromptDefault,
    _makeToolDevStartPromptDefault,
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
        developer_start_prompt: Callable = None,
        reviewer_start_prompt: Callable = None,
        step_retry_attempts: int = 5,
        max_tool_make_iterations: int = 20,
        logger: Logger = None,
        external_context_provider: Callable[[], [Dict[str,str]]] | None = None
    ) -> None:
        self.logger = logger

        # import all the external tools
        self.tools_path = tools_path
        self.tools = {}
        self._import_tools()

        # setting the default prompts
        self.make_plan_prompt = (
            make_plan_prompt if make_plan_prompt else _makePlanPromptDefault
        )
        self.take_step_prompt = (
            execution_prompt if execution_prompt else _takeActionStepPromptDefault
        )

        self.task_start_prompt = (
            task_start_prompt if task_start_prompt else _taskStartPromptDefault
        )

        self.make_tool_summerize_history_prompt = (
            task_start_prompt if task_start_prompt else _makeToolSummerizeHistoryPromptDefault
        )

        self.developer_start_prompt = (
            developer_start_prompt if developer_start_prompt else _makeToolDevStartPromptDefault
        )
        self.reviewer_start_prompt = (
            reviewer_start_prompt if reviewer_start_prompt else _makeToolReviewerStartPromptDefault
        )

        self.max_tool_make_iterations = max_tool_make_iterations

        self._llmExecutorClient = llm_client_executer
        self._llmDevClient = llm_client_dev
        self._llmReviewerClient = llm_client_reviewer
        self.llm_interactions = llm_client_interactions(self)
        self.pythonToolMaker: Python_Tool_developer = Python_Tool_developer(
            tools_path=tools_path,
            step_retry_attempts=step_retry_attempts,
            logger=self.log,
            llm_dev_client=self._llmDevClient,
            llm_reviewer_client=self._llmReviewerClient,
            developer_start_prompt = self.developer_start_prompt,
            reviewer_start_prompt = self.reviewer_start_prompt,
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

        self.running_task = True
        """ A flag to indicate if the system is currently running a task."""

        self.external_context: Callable[[], [Dict[str,str]]] = external_context_provider

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

        # Now add the special tools
        self.tools["task_complete"] = TaskCompleteTool()
        self.tools["task_complete"].function = self._taskComplete

        self.tools["update_plan"] = updatePlan()
        self.tools["update_plan"].function = self._setPlan

    def _callTool(self, tool_name: str, tool_args: Dict[str, Any]):
        """Calls a tool with the given name and arguments."""
        try:
            return str(self.tools[tool_name].function(**tool_args)) or "Tool Called without error."
        except Exception as e:
            return "Tool call failed: " + str(e)

    def CreatePythonTool(self):
        """Creates a python tool in the tools directory."""
        # Get a summary of what tool should be made and why, then make a new
        # user message and pass it to the tool maker
        summaryResponse = self._llmExecutorClient.base_invoke(
            messages=self.executionHistory + self.make_tool_summerize_history_prompt(
                use_developer= self._llmExecutorClient.use_developer
            ),
            stream=False,
            use_tools=False
        )

        tool_name = self.pythonToolMaker.MakeTool(User_msg(
            msg=summaryResponse.choices[0].message.content,
            name='Client'
        ),
        iteration_max=self.max_tool_make_iterations
        )

        # this will parse all tools and find the required modules (In theory)
        self.pythonToolMaker.install_required_modules()

        # add the tool to the list of tools
        module = self._import_tool(self.tools_path, tool_name)
        self.tools[tool_name] = module.ToolDescription()
        self.tools[tool_name].function = module.Run
        self.log(f"Tool {tool_name} created successfully.")

    def _setPlan(self, plan: str):
        """Sets the current plan to the given plan."""
        self.currentPlan = plan
        return f"Plan has been updated."

    def _makePlan(self, goal: str):
        """Make a step by step plan for the system to follow. If the goal parameter is empty then it just prompts from execution history"""

        self.executionHistory.append(User_msg(msg=goal, name='Jed'))
        self.executionHistory += self.make_plan_prompt(
            name='System',
            use_developer= self._llmExecutorClient.use_developer
            )

        
        response = self._llmExecutorClient.base_invoke(
            messages=self.executionHistory,
            stream=False,
            use_tools=True,
            tools=self.tools,
            reasoning_effort="high" if (self._llmExecutorClient.model.__contains__("o1") or self._llmExecutorClient.model.__contains__("o3")) else None
        )

        self.currentPlan = self.llm_interactions._getLLMResponseMessage(response).content
        self.executionHistory.append(Assistant_msg(
            msg=self.currentPlan,
            name='Prometheus',
            tool_calls= response.choices[0].message.tool_calls
        ))
        if response.choices[0].message.tool_calls:
            for tool in response.choices[0].message.tool_calls:
                args = self.llm_interactions._getLLMToolCall(tool)
                toolResponse = self._callTool(args[0], args[1])
                self.executionHistory.append(Tool_response(
                    call_id=tool.id,
                    content=toolResponse
                ))
        self.log(f"Plan created successfully: \n{self.currentPlan}\n", level=logging_codes.PLAN_MADE.value)

    def _executeStep(self, user_task: str|None = None):
        """Have the LLM take a step through it's plan, if a task is provided
        A new plan will be made and added to the execution history."""

        # Note anything within "Quotes" is just seen as data byt the LLM, it will
        # be untrusted data. Tool responses should probable be wrapped in this.

        if self.executionHistory.__len__() == 0:
            self.executionHistory.append(self.task_start_prompt(
                use_developer= (self._llmExecutorClient.use_developer) # this only makes sense for o1 o3-mini models
            ))

        if user_task is not None:
            # Tell the LLM to plan for the user's task
            self._makePlan(user_task)

        # Inject the external context if the function is provided at init
        context_msg = []
        if self.external_context is not None:
            context_msg = self.external_context()
            if type(context_msg) is list:
                pass
            elif type(context_msg) is dict:
                context_msg = [context_msg]
            else:
                self.log("External context provider must return a list or dictionary of messages.", level=logging_codes.ERROR.value)
                raise ValueError("External context provider must return a list or dictionary of messages.")

        # contruct prompt
        prompt = self.executionHistory + [System_msg(
                    msg=self.currentPlan,
                    name='Prometheus'
                )] + context_msg + self.take_step_prompt(
                    use_developer= (self._llmExecutorClient.use_developer) # this only makes sense for o1 o3-mini models
                )

        # Get the LLM to take a step
        stepResponse = self._llmExecutorClient.base_invoke(
            messages=prompt,
            stream=False,
            use_tools=True,
            tools=self.tools,
            # reasoning_effort="high" if (self._llmExecutorClient.model.__contains__("o1") or self._llmExecutorClient.model.__contains__("o3-mini")) else None
        )

        if context_msg is not None:
            self.executionHistory += context_msg

        self.executionHistory.append(Assistant_msg(
            msg= stepResponse.choices[0].message.content,
            name='Prometheus',
            tool_calls= stepResponse.choices[0].message.tool_calls
        ))
        if stepResponse.choices[0].message.content is not None:
            self.log(f"{stepResponse.choices[0].message.content}",
                    level=logging_codes.ACTION_COMPLETE.value)

        # make the tool calls and add the responses to the execution history
        if stepResponse.choices[0].message.tool_calls:
            for toolCall in stepResponse.choices[0].message.tool_calls:
                args = self.llm_interactions._getLLMToolCall(toolCall)
                toolResponse = self._callTool(args[0], args[1])
                self.executionHistory.append(Tool_response(
                    call_id=toolCall.id,
                    content=toolResponse
                ))

    def generate_summary(self):
        """Generate a summary of the system's actions."""

        messages = filter_system_messages(self.executionHistory) + [System_msg(
            msg=f"""You are an AI responsible for generating a summary of the system's actions.

Generate a summary of the system's actions to who informed the system to: {self.goal}.""",
            use_developer=self._llmExecutorClient.use_developer,
        )]

        return self.llm_interactions._getLLMResponseMessage(
            self._llmExecutorClient.base_invoke(
                messages=messages,
                stream=False,
                use_tools=False,
            )
        )

    def _taskComplete(self):
        """To be called by the LLM when the task is complete."""
        self.running_task = False

    def Task(self, goal: str):
        """Perform a task given by the user."""
        self.goal = goal
        self.running_task = True

        # make a plan to achieve the goal
        self._makePlan(goal)

        # execute the plan
        while self.running_task:
            self._executeStep()
        else:
            self.log("Task complete.", level=logging_codes.TASK_COMPLETE.value)
            self.log("\n" + self.generate_summary().content, level=logging_codes.TASK_COMPLETE.value)
