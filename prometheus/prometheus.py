from typing import Callable, List, Dict, Any
from os import path, listdir
from logging import Logger, INFO
import importlib.util
from prometheus.tools.definitions import *
from prometheus.utils import llm_client_interactions, logging_codes, llm_client_base
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

        self.planQueue: List[instruction] = []
        """ The queue of instructions to be executed by the system."""

        self.goal = ""
        """ The goal given to the system by the user (the main task)."""

        self.executionHistory: List[InstructionResponse] = []
        """ The history of the instructions given to the system and the responses (carries forward context)."""

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
        plan_prompt = self.make_plan_prompt(goal, self.planQueue, self.tools)

        # Invoke the LLM with a special tool for making a plan
        response = self._llmDevClient.base_invoke(
            messages=plan_prompt,
            stream=False,
            use_tools=True,
            tools={"make_plan": MakePlanTool()},
            forced_tool="make_plan",
        )

        try:
            tools = self.llm_interactions._getLLMResponseTools(response)
            for step in self.llm_interactions._getLLMToolCall(tools[0])[1]["steps"]:
                self.planQueue.append(instruction(step["action"], step["reason"]))
            self.log(
                f"Plan made: {self.planQueue}", level=logging_codes.PLAN_MADE.value
            )
        except:
            raise RuntimeError("Failed to make a plan.")


    def _executeStep(self):
        """Get the next step in the plan, determain if the system is able to do
        it and if so do it, otherwise figure out what tools are needed for this
        task and instruct the dev team to make them."""

        # get the next step in the plan
        step = self.planQueue[0]

        # ask the system if it is able to do this with the tools avaliable
        prompt = [
            {
                "role": "system",
                "content": f"You are part of a task completion system. You are to state if the follow step in a plan can be completed: \n {step.action} \n Reason: {step.reason}",
            },
            {
                "role": "system",
                "content": f"Is the system able to complete the step with the given tools?",
            },
            {
                "role": "system",
                "content": f"The tools available to the system are: \n{"\n".join([f"{tool} : {self.tools[tool].description}" for tool in self.tools.keys()])}.",
            },
            {
                "role": "system",
                "content": f"The system cannot do anything outside of the tools it has available.",
            },
            {
                "role": "system",
                "content": f"Here is the execution history of the system: \n{[f'Action:{x.action}\nResponse:{x.response}\n\n' for x in self.executionHistory]}",
            },
        ]

        response = self._llmExecutorClient.force_function_call_invoke(
            messages=prompt,
            forced_tool="can_do_step",
            tools={
                "can_do_step": CanDoStepTool()
            },
            stream=False,
        )

        # get the response from the system and the reason for thinking
        # response = self._getLLMToolCall(self.llm_interactions._getLLMResponseTools(response)[0])[1]
        can_do_step = response["can_do_step"]
        how = response["how"]

        if can_do_step:
            self.log(
                f"Prometheus thinks they can do: ({step.action}) by: ({how}). Attempting action.",
                level=logging_codes.THINKING.value,
            )

            for i in range(self.step_retry_attempts):
                # attempt the action
                self.log(
                    f"Attempting action: {step.action}. Attempt: {i+1}",
                    level=logging_codes.THINKING.value,
                )
                actionStep = self._llmExecutorClient.base_invoke(
                    messages=self.take_step_prompt(
                        step, how, self.executionHistory, self.goal
                    ),
                    stream=False,
                    use_tools=True,
                    tools=self.tools,
                )

                # check to see if the response is a tool call
                if self.llm_interactions._LLMResponseIsToolCall(actionStep):
                    # Call the tool
                    toolName, toolArgs = self.llm_interactions._getLLMToolCall(
                        self.llm_interactions._getLLMResponseTools(actionStep)[0]
                    )
                    toolResult = self._callTool(toolName, toolArgs)

                    # # get the generated response and add it to the execution history
                    self.executionHistory.append(
                        InstructionResponse(step.action, str(toolResult))
                    )
                    self.log(
                        f"Action complete: {step.action}. With result: {str(toolResult)}",
                        level=logging_codes.ACTION_COMPLETE.value,
                    )
                    # TODO see about changing the plan as the system may have learned something new
                    # self._makePlan(self.goal, self.executionHistory[-1])
                    break

                else:
                    # assume the action failed
                    if i < self.step_retry_attempts - 1:
                        self.log(
                            f"Action failed: {step.action}. trying again...",
                            level=logging_codes.ACTION_FAILED.value,
                        )
                    else:
                        self.log(
                            f"Action failed: {step.action}. No more attempts.",
                            level=logging_codes.ACTION_FAILED.value,
                        )
                        raise RuntimeError(
                            f"Action failed: {step.action}. No more attempts."
                        )
            self.planQueue.pop(0)

        # if the system cannot do the step make a tool to do it
        else:
            self.log(
                f"Prometheus thinks they cannot do: ({step.action}) because: ({how}). Making a tool to do it.",
                level=logging_codes.THINKING.value,
            )

            toolDescriptionResponse = None
            for i in range(self.step_retry_attempts * 2):

                # get the system to make a description of the tool that allows
                # the system to complete the step using the dev llm.
                response = self._llmDevClient.base_invoke(
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are an AI responsible for making tools for a system.",
                        },
                        {
                            "role": "system",
                            "content": f"The system has the step: ({step.action}) but thinks it can't do this because: ({how}).",
                        },
                        {
                            "role": "system",
                            "content": f"Make a description of a tool so the system is able to complete this step by using the 'make_tool' tool.",
                        },
                    ],
                    stream=False,
                    use_tools=True,
                    tools={
                        "make_tool": MakePythonToolTool(),
                    },
                    forced_tool="make_tool",
                )

                try:
                    toolDescriptionResponse = self.llm_interactions._getLLMToolCall(
                        self.llm_interactions._getLLMResponseTools(response)[0]
                    )[1]
                    break
                except:
                    if i < self.step_retry_attempts - 1:
                        self.log(
                            f"Failed to make a tool to complete the step: <{step.action}>. trying again...",
                            level=logging_codes.ACTION_FAILED.value,
                        )
                        continue
                    else:
                        raise RuntimeError(
                            f"Failed to make a tool to complete the step: {step.action}. No more attempts."
                        )

            # put the parameters in the correct format
            tool_parameters = []
            for parameter in toolDescriptionResponse["tool_parameters"]:

                # make sure the parameters have been correctly formatted
                if not parameter.keys() == {"name", "type", "description"}:
                    raise ValueError(' LLM response for tool parameters is not correctly formatted. Expected: {"name", "type", "description"}', parameter)
                if not parameter["type"] in ["str", "int", "list", "float", "bool"]:
                    raise ValueError(f"LLM response for tool parameters has an invalid type: {parameter['type']}")

                tool_parameters.append(
                    LLMToolParameter(
                        name=parameter['name'],
                        type=parameter['type'],
                        description=parameter['description'],
                    )
                )

            # make the tool described
            self.CreatePythonTool(
                toolDescriptionResponse["tool_name"],
                toolDescriptionResponse["tool_description"],
                tool_parameters,
                toolDescriptionResponse["tool_required_parameters"],
                toolDescriptionResponse["dev_comment"],
            )

    def generate_summary(self):
        """Generate a summary of the system's actions."""

        messages = [
            {
                "role": "system",
                "content": f"You are an AI responsible for generating a summary of the system's actions.",
            },
            {
                "role": "system",
                "content": f"The system has completed the following actions:\n {'\n'.join([f'Action: {x.action}\nResponse:{x.response}' for x in self.executionHistory])}.",
            },
            {
                "role": "system",
                "content": f"Generate a summary of the system's actions to respond to the user who gave the task: {self.goal}.",
            },
        ]

        return self.llm_interactions._getLLMResponseCompleteResponse(
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
        while self.planQueue.__len__() > 0:
            self._executeStep()
        else:
            self.log("Task complete.\n", level=logging_codes.TASK_COMPLETE.value)
            self.log(self.generate_summary(), level=logging_codes.TASK_COMPLETE.value)
