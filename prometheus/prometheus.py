import json
import dataclasses

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
class promptMessage:
    role: Literal["system", "user"]
    content: str

@dataclasses.dataclass
class instruction:
    """ An intruction given from one part of the system to another."""
    task: str
    reason: str

class Prometheus:
    """
    Self tooling LLM agent.
    """
    def __init__(self, 
                 openAI_client: OpenAI,
                 model: str,
                 tools: Dict[str, LLMTool],
                 tools_path: str = "./tools", # Path to the tools directory
                 create_tool_prompt: Callable = None,
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

        if create_tool_prompt:
            self.create_tool_prompt = create_tool_prompt
        else:
            self.create_tool_prompt = self._deault_create_tool_prompt

    def log(self, message: str, level: int = INFO):
        """ Logs a message to the logger."""
        if self.logger:
            self.logger.log(level, message)

    def GetModels(self):
        """ Returns the models available in the OpenAI API."""
        return [x["id"] for x in self._client.models.list().to_dict()["data"]]
    
    def _deault_create_tool_prompt(self, 
                                   tool_name: str, 
                                   tool_description: str, 
                                   tool_parameters: List[LLMToolParameter],
                                   tool_required_parameters: List[str],
                                   instruction: instruction):
        """ Default prompt to create a tool."""
        return [
            {"role": "system", "content": f"You are an experienced python AI programmer who has been tasked with creating a python script called {tool_name} with the description: {tool_description}, for part of a larger system."},
            {"role": "system", "content": f"You have been given the following Instruction: \n {instruction.task} \n Reason: {instruction.reason}"},
            {"role": "system", "content": f"The following are required parameters: {', '.join(tool_required_parameters)}. The rest are optional."},
            {"role": "system", "content": f"Create a single python script. "},
            {"role": "system", "content": f"The script must have a global function called 'Run' so it can be called by the system. "},
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
        """ Returns the tool used in the response from the LLM 
        (if it decided to use a tool)."""
        return LLM_response.choices[0].message.tool_calls
    
    def _getLLMToolCall(self, LLM_response_tool):
        """ Returns a tuple of the name of the tool called and it's arguments.
        (An item from _getLLMResponseTools)"""
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
                raise ValueError("The forced tool is not in the tools list.")

        self.log(f"Invoking the LLM API with messages: {messages}")

        response = self._client.chat.completions.create(
            model=self.active_model,
            messages=messages,
            stream=stream,
            tools = self._getFormattedTools(tools) if use_tools else [],
            tool_choice = self._formatToolChoice(forced_tool) if forced_tool else None,
        )
        return response
    
    def _callTool(self, tool_name: str, tool_args: Dict[str, Any]):
        """ Calls a tool with the given name and arguments."""
        return self.tools[tool_name].function(**tool_args)
    
    def MakeTool(self,
                 tool_name: str,
                 tool_description: str,
                 tool_parameters: List[LLMToolParameter],
                 tool_required_parameters: List[str],
                 instruction: instruction
                 ):
        """Creates a python tool in the tools directory."""
        # Create the tool prompt
        tool_prompt = self.create_tool_prompt(tool_name, tool_description, tool_parameters, tool_required_parameters, instruction)

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

        # prepend the imports and the ToolDescription function
        pythonCode = f"""from prometheus import LLMTool, LLMToolParameter

{pythonCode}

def ToolDescription():
    return LLMTool(
        name="{tool_name}",
        description="{tool_description}",
        parameters={[f"LLMToolParameter({parameter.name, parameter.type, parameter.description})" for parameter in tool_parameters]},
        requiredParameters={[tool_required_parameters]},
        type="function"
    )
"""
        
        # Writing the python file to the tools directory
        with open(path.join(self.tools_path, f"{tool_name}.py"), "w") as f:
            f.write(pythonCode)
        
        # now import the tool
        module = self._import_tool(self.tools_path, tool_name)

        # add the tool to the list of tools
        self.tools[tool_name] = module.ToolDescription()
        self.tools[tool_name].function = module.Run
        self.log(f"Tool {tool_name} created successfully.")


if __name__ == "__main__":
    prometheus = Prometheus(
        openAI_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="None" # Required by the class but not used by ollama
        ),
        model="qwen2.5:latest",
        tools={},
        tools_path="./tools"
    )

    prometheus.MakeTool(
        tool_name="change_desktop_wallpaper",
        tool_description="change the desktop wallpaper for the user",
        tool_parameters=[ 
            LLMToolParameter(name="image_path", type="str", description="The path to the image to set the wallpaper to.")
        ],
        tool_required_parameters=["image_path"],
        instruction=instruction(
            task="Create a python script that changes the desktop wallpaper on a arch linux wayland system using kde plasma.",
            reason="To provide a programatic way to change the desktop wallpaper."
        )
    )