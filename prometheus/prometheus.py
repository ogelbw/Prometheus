import json
import logging

from openai import OpenAI
from typing import Callable, List, Literal, Dict, Any
from os import path, listdir
from logging import getLogger, Logger, INFO, WARNING, ERROR, CRITICAL
import importlib.util
import re
from prometheus.tools.definitions import *
from prometheus.utils import llm_client_interactions, logging_codes, llm_client_base
from prometheus.devTeam.toolCreator import Python_Tool_developer

class Prometheus:
    """
    Self tooling LLM agent.
    """
    def __init__(self,
                 llm_client: llm_client_base,
                 tools_path: str = "./tools", # Path to the tools directory
                 make_tool_prompt: Callable = None,
                 make_plan_prompt: Callable = None,
                 execution_prompt: Callable = None,
                 step_retry_attempts: int = 5,
                 logger: Logger = None,
                 dev_llm_model: str = "qwen2.5:latest",
                 reviewer_llm_model: str = "qwen2.5:latest"
                 ) -> None:
        self._client = llm_client
        self.logger = logger
        self.llm_interactions = llm_client_interactions(self)
        self.pythonToolMaker = Python_Tool_developer(tools_path=tools_path, 
                                                     step_retry_attempts=step_retry_attempts, 
                                                     logger=self.log,
                                                     dev_llm_model=dev_llm_model,
                                                     reviewer_llm_model=reviewer_llm_model)

        # Check if the model is available
        if dev_llm_model not in self._client.GetModels():
            raise ValueError("dev Model not available.")
        elif reviewer_llm_model not in self._client.GetModels():
            raise ValueError("reviewer Model not available.")

        # import all the external tools
        self.tools_path = tools_path
        self.tools = {}
        self._import_tools()

        # set the constructor for the prompt for creating python tools
        if make_tool_prompt:
            self.create_tool_prompt = make_tool_prompt
        else:
            self.create_tool_prompt = self._makeToolPromptDefault

        # set the constructor for the prompt for evaluating the plan
        if make_plan_prompt:
            self.make_plan_prompt = make_plan_prompt
        else:
            self.make_plan_prompt = self._makePlanPromptDefault

        # set the constructor for the prompt for evaluating the plan
        if make_plan_prompt:
            self.update_plan_prompt = make_plan_prompt
        else:
            self.update_plan_prompt = self._updatePlanPromptDefault

        # set the constructor for the prompt for taking a step
        if execution_prompt:
            self.take_step_prompt = execution_prompt
        else:
            self.take_step_prompt = self._takeActionStepPromptDefault

        self.planQueue:List[instruction] = []
        """ The queue of instructions to be executed by the system."""

        self.goal = ""
        """ The goal given to the system by the user (the main task)."""

        self.executionHistory:List[InstructionResponse] = []
        """ The history of the instructions given to the system and the responses (carries forward context)."""

        self.step_retry_attempts = step_retry_attempts
        """ The number of times to retry a step if it fails."""

    def log(self, message: str, level: int = INFO):
        """ Logs a message to the logger."""
        if self.logger:
            self.logger.log(level, message)

    def _makeToolPromptDefault(self,
                               tool_name: str,
                               tool_description: str,
                               tool_parameters: List[LLMToolParameter],
                               tool_required_parameters: List[str],
                               comment: str):
        """ Default prompt to create a tool."""
        return [
            {"role": "system", "content": f"You are an experienced python AI programmer who has been tasked with creating a python script called {tool_name} with the description: {tool_description}, for part of a larger system."},
            {"role": "system", "content": f"The tool must have the following parameters: {', '.join([f'{x.name} ({x.type}) : {x.description}' for x in tool_parameters])}."},
            {"role": "system", "content": f"The following are required parameters: {', '.join(tool_required_parameters)}. The rest are optional."},
            {"role": "system", "content": f"Create a single python script. "},
            {"role": "system", "content": f"The script must have a global function called 'Run' so it can be called by the system INCLUDE THIS FUNCTION."},
            {"role": "system", "content": f"Additionally this comment to the developer was left: {comment}"},
        ]

    def _makePlanPromptDefault(self, final_goal: str, plan: List[instruction]):
        return [
            {"role": "system", "content": f"You are a AI planner part of a larger system. This system can make it's own tools in python and is already logged into the user's machine."},
            {"role": "system", "content": f"The user gave the system this goal: <{final_goal}>."},
            {"role": "system", "content": f"This is the current plan to achieve the goal:\n{[f"{i}. {x.action}, Reason: {x.reason}" for i, x in enumerate(plan)]}"},
            {"role": "system", "content": f"The system has access to these tools: [{', '.join(self.tools.keys())}]"},
            {"role": "system", "content": f"The system can create new tools in python."},
            {"role": "system", "content": f"Create a plan to achieve the user's goal."},
        ]

    def _updatePlanPromptDefault(self, final_goal: str, plan: List[instruction], previous_action_response: InstructionResponse):
        return [
            {"role": "system", "content": f"You are a AI planner part of a larger system. This system can make it's own tools in python and is already logged into the user's machine."},
            {"role": "system", "content": f"The user gave the system this goal: <{final_goal}>."},
            {"role": "system", "content": f"This is the current plan to achieve the goal:\n{[f"{i}. {x.action}, Reason: {x.reason}" for i, x in enumerate(plan)]}"},
            {"role": "system", "content": f"The system has access to these tools: [{', '.join(self.tools.keys())}]"},
            {"role": "system", "content": f"The systems previous action was: {previous_action_response.action} with the result: {previous_action_response.response}."},
            {"role": "system", "content": f"Update the plan (by making a new one) to capture the result of the previous action."},
        ]

    def _takeActionStepPromptDefault(self, step: instruction, how: str):
        return [
            {"role": "system", "content": f"You are an AI responsible for taking actions in a system."},
            {"role": "system", "content": f"Here is the previous execution history of the system: {". ".join([x.response for x in self.executionHistory])}"},
            {"role": "system", "content": f"The user gave the system this goal: <{self.goal}>."},
            {"role": "system", "content": f"The current step is to: <{step.action}> because: <{step.reason}>."},
            {"role": "system", "content": f"The system thinks this can be done by: <{how}>."},
            {"role": "system", "content": f"Do this step as described."},
        ]

    def _import_tool(self, tools_path: str, tool_name: str):
        """ Imports a tool from the tools directory as a module."""

        # loading the python file as a module
        spec = importlib.util.spec_from_file_location(tool_name, path.join(tools_path, f"{tool_name}.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def _import_tools(self):
        """ Imports all of the tools in the tools directory."""
        tools = {}
        tools_path = path.abspath(self.tools_path)
        for tool_name in [x.split(".")[0] for x in listdir(tools_path) if x.endswith(".py")]:
            tools[tool_name] = self._import_tool(tools_path, tool_name)

            # add the tool to the tools dictionary using the ToolDescription
            # function in all tool scripts
            self.tools[tool_name] = tools[tool_name].ToolDescription()
            self.tools[tool_name].function = tools[tool_name].Run
            self.log(f"Tool {tool_name} imported successfully.")

    def _callTool(self, tool_name: str, tool_args: Dict[str, Any]):
        """ Calls a tool with the given name and arguments."""
        return self.tools[tool_name].function(**tool_args)

    def MakeTool(self,
                 tool_name: str,
                 tool_description: str,
                 tool_parameters: List[LLMToolParameter],
                 tool_required_parameters: List[str],
                 developer_comment: str
                 ):
        """Creates a python tool in the tools directory."""
        self.pythonToolMaker.MakeTool(tool_name,
                                      tool_description,
                                      tool_parameters,
                                      tool_required_parameters,
                                      developer_comment)

        # add the tool to the list of tools
        module = self._import_tool(self.tools_path, tool_name)
        self.tools[tool_name] = module.ToolDescription()
        self.tools[tool_name].function = module.Run
        self.log(f"Tool {tool_name} created successfully.")

    def _makePlan(self, goal:str):
        """ Make a step by step plan for the system to follow."""
        plan_prompt = self.make_plan_prompt(goal, self.planQueue)

        # Invoke the LLM with a special tool for making a plan
        response = self._client.base_invoke(
            model=self.pythonToolMaker.dev_llm_model,
            messages=plan_prompt,
            stream=False,
            use_tools=True,
            tools= {"make_plan": LLMTool(
                name="make_plan",
                description="Make a step by step plan for the system to follow.",
                requiredParameters=["steps"],
                type="function",
                parameters=[LLMToolParameter(
                    name="steps",
                    type="List[Dict[str, str]]",
                    description="The steps to be included in the plan in the format {action : str, reason : str}.",
            )])},
            forced_tool="make_plan",
        )

        try:
            tools = self.llm_interactions._getLLMResponseTools(response)
            for step in self.llm_interactions._getLLMToolCall(tools[0])[1]['steps']:
                self.planQueue.append(instruction(step['action'], step['reason']))
            self.log(f"Plan made: {self.planQueue}", level=logging_codes.PLAN_MADE.value)
        except:
            raise RuntimeError("Failed to make a plan.")

    def _makePlan(self, goal:str, previous_action_response: InstructionResponse | None = None):
        """ Make a step by step plan for the system to follow."""
        if previous_action_response:
            plan_prompt = self.update_plan_prompt(goal, self.planQueue, previous_action_response)
        else:
            plan_prompt = self.make_plan_prompt(goal, self.planQueue)

        for i in range(self.step_retry_attempts):
            # Invoke the LLM with a special tool for making a plan
            response = self._client.base_invoke(
                model=self.pythonToolMaker.reviewer_llm_model,
                messages=plan_prompt,
                stream=False,
                use_tools=True,
                tools= {"make_plan": LLMTool(
                    name="make_plan",
                    description="Make a step by step plan for the system to follow.",
                    requiredParameters=["steps"],
                    type="function",
                    parameters=[LLMToolParameter(
                        name="steps",
                        type="List[Dict[str, str]]",
                        description="The steps to be included in the plan in the format {action : str, reason : str}.",
                )])},
                forced_tool="make_plan",
            )

            try:
                tools = self.llm_interactions._getLLMResponseTools(response)
                for step in self.llm_interactions._getLLMToolCall(tools[0])[1]['steps']:
                    self.planQueue.append(instruction(step['action'], step['reason']))

                self.log(f"Plan made: {self.planQueue}", level=logging_codes.PLAN_MADE.value)
                break
            except:
                if i < self.step_retry_attempts - 1:
                    self.log(f"Failed to make a plan. Trying again...", level=logging_codes.PLAN_FAILED.value)
                else:
                    raise RuntimeError("Failed to make a plan.")

    def _executeStep(self):
        """ Get the next step in the plan, determain if the system is able to do
        it and if so do it, otherwise figureout what tools are needed for this
        task and instruct the dev team to make them."""

        # get the next step in the plan
        step = self.planQueue[0]

        # ask the system if it is able to do this with the tools avaliable
        prompt = [
                {"role": "system", "content": f"You are part of a task completion system. You are to state if the follow step in a plan can be completed: \n {step.action} \n Reason: {step.reason}"},
                {"role": "system", "content": f"Is the system able to complete the step with the given tools?"},
                {"role": "system", "content": f"The tools available to the system are: \n{"\n".join([f"{tool} : {self.tools[tool].description}" for tool in self.tools.keys()])}."},
                {"role": "system", "content": f"The system cannot do anything outside of the tools it has available."},
                {"role": "system", "content": f"Here is the execution history of the system: \n{[f'{x.action} : {x.response}' for x in self.executionHistory]}"},
            ]

        response = self._client.force_function_call_invoke(
            model=self.pythonToolMaker.reviewer_llm_model,
            messages=prompt,
            forced_tool="can_do_step",
            tools={"can_do_step" : LLMTool(
                name="can_do_step",
                description="Determine if the system can complete the step with the tools available.",
                requiredParameters=["can_do_step", "how"],
                type="function",
                parameters=[
                    LLMToolParameter(
                        name="can_do_step",
                        type="bool",
                        description="A boolean value indicating if the system can complete the step.",
                    ),
                    LLMToolParameter(
                        name="how",
                        type="str",
                        description="A description of how to complete the step or a reason it cannot be completed.",
                    ),
                ]
            )},
            stream=False)

        # get the response from the system and the reason for thinking
        # response = self._getLLMToolCall(self.llm_interactions._getLLMResponseTools(response)[0])[1]
        can_do_step = response['can_do_step']
        how = response['how']

        if can_do_step:
            self.log(f"Prometheus thinks they can do: ({step.action}) by: ({how}). Attempting action.", level=logging_codes.THINKING.value)

            for i in range(self.step_retry_attempts):
                # attempt the action
                self.log(f"Attempting action: {step.action}. Attempt: {i+1}", level=logging_codes.THINKING.value)
                actionStep = self._client.base_invoke(
                    model=self.pythonToolMaker.reviewer_llm_model,
                    messages=self.take_step_prompt(step, how),
                    stream=False,
                    use_tools=True,
                    tools=self.tools
                )

                # check to see if the response is a tool call
                if self.llm_interactions._LLMResponseIsToolCall(actionStep):
                    # Call the tool
                    toolName, toolArgs = self.llm_interactions._getLLMToolCall(self.llm_interactions._getLLMResponseTools(actionStep)[0])
                    toolResult = self._callTool(toolName, toolArgs)

                    # # get the generated response and add it to the execution history
                    self.executionHistory.append(InstructionResponse(step.action, str(toolResult)))
                    self.log(f"Action complete: {step.action}. With result: {str(toolResult)}", level=logging_codes.ACTION_COMPLETE.value)
                    # TODO see about changing the plan as the system may have learned something new
                    # self._makePlan(self.goal, self.executionHistory[-1])
                    break

                else:
                    # assume the action failed
                    if i < self.step_retry_attempts - 1:
                        self.log(f"Action failed: {step.action}. trying again...", level=logging_codes.ACTION_FAILED.value)
                    else:
                        self.log(f"Action failed: {step.action}. No more attempts.", level=logging_codes.ACTION_FAILED.value)
                        raise RuntimeError(f"Action failed: {step.action}. No more attempts.")
            self.planQueue.pop(0)

        # if the system cannot do the step make a tool to do it
        else:
            self.log(f"Prometheus thinks they cannot do: ({step.action}) because: ({how}). Making a tool to do it.", level=logging_codes.THINKING.value)

            toolDescriptionResponse = None
            for i in range(self.step_retry_attempts*2):
                # get the system to make a description of a tool to do the step
                response = self._client.base_invoke(
                    model=self.pythonToolMaker.dev_llm_model,
                    messages=[
                        {"role": "system", "content": f"You are an AI responsible for making tools for a system."},
                        {"role": "system", "content": f"The system has the step: ({step.action}) but thinks it can't do this because: <{how}>."},
                        {"role": "system", "content": f"Make a tool so the system is able to complete this step."},
                    ],
                    stream=False,
                    use_tools=True,
                    tools={"make_tool": LLMTool(
                        name="make_tool",
                        description="Makes a python tool for the system.",
                        requiredParameters=["tool_name", "tool_description", "tool_parameters", "tool_required_parameters", "dev_comment"],
                        type="function",
                        parameters=[
                            LLMToolParameter(name="tool_name",
                                            type="str",
                                            description="The name of the tool.",),
                            LLMToolParameter(name="tool_description",
                                            type="str",
                                            description="A description of the tool.",),
                            LLMToolParameter(name="tool_parameters",
                                            type="Dict[str, Tuple[str, str]]",
                                            description="The parameters of the tool in the format: key: parameter_name[str], value: (parameter_description[str], parameter_type[str])",),
                            LLMToolParameter(name="tool_required_parameters",
                                            type="List[str]",
                                            description="The required parameters of the tool.",),
                            LLMToolParameter(name="dev_comment",
                                            type="str",
                                            description="A comment to the developer of the tool. Any additional requirements not captured by the description.",),
                        ]
                    )},
                    forced_tool="make_tool"
                )

                # getting the description of the tool
                try:
                    toolDescriptionResponse = self.llm_interactions._getLLMToolCall(self.llm_interactions._getLLMResponseTools(response)[0])[1]
                    break
                except:
                    if i < self.step_retry_attempts - 1:
                        self.log(f"Failed to make a tool to complete the step: <{step.action}>. trying again...", level=logging_codes.ACTION_FAILED.value)
                        continue
                    else:
                        raise RuntimeError(f"Failed to make a tool to complete the step: {step.action}. No more attempts.")

            # put the parameters in the correct format
            tool_parameters = []
            for parameter in toolDescriptionResponse['tool_parameters'].keys():
                tool_parameters.append(LLMToolParameter(
                    name=parameter,
                    type=toolDescriptionResponse['tool_parameters'][parameter][1],
                    description=toolDescriptionResponse['tool_parameters'][parameter][0]
                ))

            # make the tool described
            self.MakeTool(toolDescriptionResponse['tool_name'],
                          toolDescriptionResponse['tool_description'],
                          tool_parameters,
                          toolDescriptionResponse['tool_required_parameters'],
                          toolDescriptionResponse['dev_comment'])

    def generate_summary(self):
        """ Generate a summary of the system's actions."""

        messages = [
            {"role": "system", "content": f"You are an AI responsible for generating a summary of the system's actions."},
            {"role": "system", "content": f"The system has completed the following actions:\n {'\n'.join([f'Action: {x.action}\nResponse:{x.response}' for x in self.executionHistory])}."},
            {"role": "system", "content": f"Generate a summary of the system's actions to respond to the user who gave the task: {self.goal}."},
        ]

        return self.llm_interactions._getLLMResponseCompleteResponse(
            self._client.base_invoke(
                model=self.pythonToolMaker.reviewer_llm_model,
                messages=messages,
                stream=False,
                use_tools=False,
            )
        )

    def Task(self, goal:str):
        """ Perform a task given by the user."""
        self.goal = goal

        # make a plan to achieve the goal
        self._makePlan(goal)

        # execute the plan
        while self.planQueue.__len__() > 0:
            self._executeStep()
        else:
            self.log("Task complete.\n", level=logging_codes.TASK_COMPLETE.value)
            self.log(self.generate_summary(), level=logging_codes.TASK_COMPLETE.value)