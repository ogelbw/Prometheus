import json
import dataclasses
import logging

from openai import OpenAI
from typing import Callable, List, Literal, Dict, Any
from os import path, listdir
from logging import getLogger, Logger, INFO, WARNING, ERROR, CRITICAL
import importlib.util
import re

@dataclasses.dataclass
class LLMToolParameter:
    name: str
    type: str
    description: str

@dataclasses.dataclass
class LLMTool:
    name: str
    description: str
    parameters: List[LLMToolParameter]
    requiredParameters: List[str]
    type: Literal["function", "data"]
    function: Callable|None = None
    asyncFunction: bool = False

@dataclasses.dataclass
class instruction:
    """ An intruction given from one part of the system to another."""
    action: str
    reason: str

@dataclasses.dataclass
class InstructionResponse:
    """ A response to an instruction."""
    action: str
    response: str

class Prometheus:
    """
    Self tooling LLM agent.
    """
    
    TOOL_MADE = 50
    TOOL_USED = 51
    
    THINKING = 60
    ACTION_COMPLETE = 61
    ACTION_FAILED = 62

    PLAN_MADE = 70
    PLAN_FAILED = 71

    def __init__(self, 
                 openAI_client: OpenAI,
                 model: str,
                 tools: Dict[str, LLMTool],
                 tools_path: str = "./tools", # Path to the tools directory
                 make_tool_prompt: Callable = None,
                 make_plan_prompt: Callable = None,
                 execution_prompt: Callable = None,
                 step_retry_attempts: int = 5,
                 logger: Logger = None
                 ) -> None:
        self.tools = tools
        self._client = openAI_client
        self.logger = logger

        # Check if the model is available
        if model not in self.GetModels():
            raise ValueError("Model not available.")
        self.active_model = model
        

        # import all the external tools
        self.tools_path = tools_path
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
            self.take_step_prompt = self._takeStepPromptDefault

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

    def GetModels(self):
        """ Returns the models available in the OpenAI API."""
        return [x["id"] for x in self._client.models.list().to_dict()["data"]]
    
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
    
    def _takeStepPromptDefault(self, step: instruction, how: str):
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

    def _getLLMResponseTools(self, LLM_response) -> List[Any]:
        """ Returns the tools used in the response from the LLM 
        (if it decided to use a tool)."""
        return LLM_response.choices[0].message.tool_calls
    
    def _getLLMToolCall(self, LLM_response_tool):
        """ Returns a tuple of the name of the tool called and it's arguments.
        (An item from _getLLMResponseTools)"""
        self.log(f"Tool call: {LLM_response_tool.function.name} with arguments: {json.loads(LLM_response_tool.function.arguments)}", level=self.TOOL_USED)
        return (LLM_response_tool.function.name, json.loads(LLM_response_tool.function.arguments))
    
    def _getFormattedTools(self, tools:Dict[str, LLMTool] = None):
        """ Formats the tools dictionary into a list of tools that can be passed to
        the OpenAI API."""
        tmp = []

        tools = tools if tools else self.tools
        for key, tool in tools.items():
            tmp.append({
                "type": "function",
                "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {parameter.name: {
                        "type": parameter.type,
                        "description": parameter.description
                    } for parameter in tool.parameters},
                    "required": tool.requiredParameters,
                },
            }})

            # The api doesn't understand "data" type. I am using it as an indicator 
            # for myself when we want a data in a json format. To do this we pretend
            # that a function exists that uses parameters which are the data we want.
            if tool.type == "data":
                tmp[-1]["type"] = "function"
        return tmp

    def _formatToolChoice(self, tool_choice:str):
        """ Formats a name of a tool into a dictionary that can be passed to the 
        OpenAI API."""
        return  {"type": "function", "function": {"name": tool_choice}}

    def _getResponseStopReason(self, LLM_response):
        """ Returns the reason the response from the LLM stopped."""
        return LLM_response.choices[0].finish_reason

    def _getLLMResponseDelta(self, LLM_response):
        """ Returns the delta of the response from the LLM when the response is
        streamed"""
        return LLM_response.choices[0].delta.content

    def _getLLMResponseCompleteResponse(self, LLM_response):
        """ Returns the complete response from the LLM when the response is not
        streamed"""
        return LLM_response.choices[0].message.content
    
    def _LLMResponseIsToolCall(self, LLM_response):
        """ Returns whether the response from the LLM is a tool call."""
        return self._getResponseStopReason(LLM_response) == "tool_calls"

    def _baseInvoke(self, 
                    messages: List[Dict[str, str]],
                    stream: bool = False, 
                    use_tools: bool = False,
                    forced_tool: str = None,
                    tools: Dict[str, LLMTool] = None,
                    ):
        """ Returns the Invoke the LLM api with the given messages."""
        tools = tools if tools else self.tools

        if use_tools and forced_tool:
            # check that the forced tool is in the tools list
            if forced_tool not in tools:
                raise ValueError("The forced tool is not in the tools list.", forced_tool, tools.keys())

        # self.log(f"Invoking the LLM API with messages: {messages}")

        response = self._client.chat.completions.create(
            model=self.active_model,
            messages=messages,
            stream=stream,
            tools = self._getFormattedTools(tools) if use_tools else [],
            tool_choice = self._formatToolChoice(forced_tool) if forced_tool else None,
        )
        return response

    def _forcedFunctionInvoke(self,
                              messages: List[Dict[str, str]],
                              forced_tool: str,
                              tools: Dict[str, LLMTool],
                              stream: bool = False,):
        """ Invokes the LLM API with a forced tool."""

        for i in range(self.step_retry_attempts):
            response = self._baseInvoke(
                messages=messages,
                stream=stream,
                use_tools=True,
                forced_tool=forced_tool,
                tools=tools
            )

            # check to see if the response is a tool call
            try:
                response = self._getLLMToolCall(self._getLLMResponseTools(response)[0])[1]
                break
            except:
                if i < self.step_retry_attempts - 1:
                    self.log(f"Failed to call tool: {forced_tool}. Trying again...", level=self.ACTION_FAILED)
                else:
                    raise RuntimeError(f"Failed to call tool: {forced_tool}. No more attempts.")
        return response

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
        # Create the tool prompt
        tool_prompt = self.create_tool_prompt(tool_name, tool_description, tool_parameters, tool_required_parameters, developer_comment)

        response = self._baseInvoke(
            messages=tool_prompt,
            stream=False,
            use_tools=False
        )

        # Note the self._LLMResponseIsToolCall won't work here because 
        # A 'function' is forced to be used. This is a thing with openAI's API
        pythonCode = ''
        completeResponse = self._getLLMResponseCompleteResponse(response)

        # the response is a complete chat response since it fails to 
        # use reponse formatting. we can use regex to get the python code
        python_chunks = re.findall(r'```python\n(.*?)\n```', completeResponse, re.DOTALL)
        if python_chunks:
            for chunk in python_chunks:
                pythonCode += f"\n{chunk}"
                break # TODO see about removing this break, currently it takes only the first python code block as the LLM has a tendeciy to return multiple code blocks

        # prepend the imports and the ToolDescription function
        pythonCode = f"""if __name__ != "__main__": from prometheus import LLMTool, LLMToolParameter

{pythonCode}

def ToolDescription():
    return LLMTool(
        name="{tool_name}",
        description="{tool_description}",
        parameters=[{",".join([f"LLMToolParameter(name='{parameter.name}', type='{parameter.type}', description='{parameter.description}')" for parameter in tool_parameters])}],
        requiredParameters={tool_required_parameters},
        type="function"
    )
"""

        # Writing the python file to the tools directory
        with open(path.join(self.tools_path, f"{tool_name}.py"), "w") as f:
            f.write(pythonCode)

        for i in range(self.step_retry_attempts):
            # get the tool to install all the needed packages
            installPackageReponse = self._baseInvoke(
                messages=[
                    {"role": "system", "content": f"You are an AI responsible for installing packages in the system."},
                    {"role": "system", "content": f"Install all the needed packages via the pip tool for the following python code."},
                    {"role": "system", "content": f"{pythonCode}"},
                ],
                stream=False,
                use_tools=True,
                tools={"pip": self.tools["pip"]},
                forced_tool="pip"
            )

            # see if the LLM tried to make use the pip tool
            try:
                packages:Dict[str, List[str]] = self._getLLMToolCall(self._getLLMResponseTools(installPackageReponse)[0])[1]
                idsToDel = []
                for package in packages['package_names']:
                    if "prometheus" in package:
                        idsToDel.append(package)
                for id in idsToDel:
                    packages['package_names'].remove(id)
                self._callTool("pip", packages)
                break
            except Exception as e:
                print(e)
                self.log(f"Failed to install packages for tool: {tool_name}.", level=self.ACTION_FAILED)
                if i < self.step_retry_attempts - 1:
                    self.log(f"Trying again...", level=self.ACTION_FAILED)
                else:
                    raise RuntimeError(f"Failed to install packages for tool: {tool_name}. No more attempts.")

        # now import the tool
        module = self._import_tool(self.tools_path, tool_name)

        # add the tool to the list of tools
        self.tools[tool_name] = module.ToolDescription()
        self.tools[tool_name].function = module.Run
        self.log(f"Tool {tool_name} created successfully.")

    def _makePlan(self, goal:str):
        """ Make a step by step plan for the system to follow."""
        plan_prompt = self.make_plan_prompt(goal, self.planQueue)

        # Invoke the LLM with a special tool for making a plan
        response = self._baseInvoke(
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
            tools = self._getLLMResponseTools(response)
            for step in self._getLLMToolCall(tools[0])[1]['steps']:
                self.planQueue.append(instruction(step['action'], step['reason']))
            
            self.log(f"Plan made: {self.planQueue}", level=self.PLAN_MADE)
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
            response = self._baseInvoke(
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
                tools = self._getLLMResponseTools(response)
                for step in self._getLLMToolCall(tools[0])[1]['steps']:
                    self.planQueue.append(instruction(step['action'], step['reason']))
                
                self.log(f"Plan made: {self.planQueue}", level=self.PLAN_MADE)
                break
            except:
                if i < self.step_retry_attempts - 1:
                    self.log(f"Failed to make a plan. Trying again...", level=self.PLAN_FAILED)
                else:
                    raise RuntimeError("Failed to make a plan.")

    def _executeStep(self):
        """ Get the next stop in the plan, determain if the system is able to do it and if so do it, 
        otherwise make the tools needed for this."""

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

        response = self._forcedFunctionInvoke(messages=prompt,
                                              forced_tool="can_do_step",
                                              tools={"can_do_step" : LLMTool(
                                                  name="can_do_step",
                                                  description="Determine if the system can complete the step with the tools available.",
                                                  requiredParameters=["can_do_step", "how"],
                                                  type="function",
                                                  parameters=[
                                                      LLMToolParameter(name="can_do_step",
                                                                      type="bool",
                                                                      description="A boolean value indicating if the system can complete the step.",),
                                                      LLMToolParameter(name="how",
                                                                      type="str",
                                                                      description="A description of how to complete the step or a reason it cannot be completed.",),
                                                  ]
                                              )},
                                              stream=False)

        # get the response from the system and the reason for thinking
        # response = self._getLLMToolCall(self._getLLMResponseTools(response)[0])[1]
        can_do_step = response['can_do_step']
        how = response['how']

        if can_do_step:
            self.log(f"Prometheus thinks they can do: ({step.action}) by: ({how}). Attempting action.", level=self.THINKING)

            for i in range(self.step_retry_attempts):
                # attempt the action
                self.log(f"Attempting action: {step.action}. Attempt: {i+1}", level=self.THINKING)
                actionStep = self._baseInvoke(
                    messages=self.take_step_prompt(step, how),
                    stream=False,
                    use_tools=True,
                )

                # check to see if the response is a tool call
                if self._LLMResponseIsToolCall(actionStep):
                    # Call the tool
                    toolName, toolArgs = self._getLLMToolCall(self._getLLMResponseTools(actionStep)[0])
                    toolResult = self._callTool(toolName, toolArgs)

                    # log the result by getting the system to write a response
                    response = self._forcedFunctionInvoke(
                        messages=[
                            {"role": "system", "content": f"You are an AI responsible for logging the results of an action."},
                            {"role": "system", "content": f"The larger system you are a part of called the tool '{toolName}' to complete the step: {step.action} because {step.reason}."},
                            {"role": "system", "content": f"The result of calling the tool was: {toolResult}."},
                            {"role": "system", "content": f"Log the action for the system using the 'log_action' tool."},
                            ],
                        stream=False,
                        tools={"log_action": LLMTool(
                            name="log_action",
                            description="Log the result of an action.",
                            requiredParameters=["result"],
                            type="function",
                            parameters=[
                                LLMToolParameter(name="result",
                                                type="str",
                                                description="The result of the action.",),
                            ]
                        )},
                        forced_tool="log_action"
                    )

                    # get the generated response and add it to the execution history
                    resultOfAction = response['result']
                    self.executionHistory.append(InstructionResponse(step.action, resultOfAction))
                    self.log(f"Action complete: {step.action}. With result: {resultOfAction}", level=self.ACTION_COMPLETE)
                    # TODO see about changing the plan as the system may have learned something new 
                    # self._makePlan(self.goal, self.executionHistory[-1])
                    break

                else:
                    # assume the action failed
                    if i < self.step_retry_attempts - 1:
                        self.log(f"Action failed: {step.action}. trying again...", level=self.ACTION_FAILED)
                    else:
                        self.log(f"Action failed: {step.action}. No more attempts.", level=self.ACTION_FAILED)
                        raise RuntimeError(f"Action failed: {step.action}. No more attempts.")
            self.planQueue.pop(0)

        # if the system cannot do the step make a tool to do it
        else:
            self.log(f"Prometheus thinks they cannot do: ({step.action}) because: ({how}). Making a tool to do it.", level=self.THINKING)

            toolDescriptionResponse = None
            for i in range(self.step_retry_attempts*2):
                # get the system to make a description of a tool to do the step
                response = self._baseInvoke(
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

                try:
                    # get the tool description from the system
                    toolDescriptionResponse = self._getLLMToolCall(self._getLLMResponseTools(response)[0])[1]
                    break
                except:
                    if i < self.step_retry_attempts - 1:
                        self.log(f"Failed to make a tool to complete the step: <{step.action}>. trying again...", level=self.ACTION_FAILED)
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

    def Task(self, goal:str):
        """ Perform a task given by the user."""
        self.goal = goal

        # make a plan to achieve the goal
        self._makePlan(goal)

        # execute the plan
        while self.planQueue.__len__() > 0:
            self._executeStep()
        else:
            self.log("Task complete.")

if __name__ == "__main__":
    # setting up logger
    logging.basicConfig(level=logging.INFO)
    logger = getLogger(__name__)

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