import dataclasses
from typing import Dict, List, Literal, Callable, Type
from pydantic import BaseModel, Field

# Messages supported by the openai API
def System_msg(msg: str, name: str = None, use_developer: bool = False):
    apiMsg = { "content": msg }
    apiMsg.update({ "name": name } if name else {})
    apiMsg.update({ "role": "developer" if use_developer else "system" })
    return apiMsg

def Tool_response(call_id: str, content: str):
    return { "role": "tool", "content": content, "tool_call_id": call_id }

def Assistant_msg(msg: str, name: str = None, tool_calls: list = []):
    apiMsg = { "content": msg, "role": "assistant"}
    apiMsg.update({ "name": name } if name else {})
    apiMsg.update({ "tool_calls": tool_calls } if tool_calls else {})
    return apiMsg

def User_msg(msg: str, name: str = None):
    apiMsg = { "content": msg, "role": "user" }
    apiMsg.update({ "name": name } if name else {})
    return apiMsg

@dataclasses.dataclass
class LLMToolParameter:
    """ The parameters for generating a tool, not to be used in the description of a tool."""
    name: str
    type: Literal["str", "list", "int", "float", "boolean", "object"]
    description: str

@dataclasses.dataclass
class LLMTool:
    """ A tool that can be used by the LLM."""
    name: str
    description: str
    parameters: BaseModel
    requiredParameters: List[str]
    type: Literal["function", "data"]
    function: Callable|None = None

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

# --- Special LLM tool calls ---
class MakePlanToolParameters(BaseModel):
    steps: list = Field(..., description="The steps for the plan in the format: [{\"action\" : str, \"reason\" : str}...]. The action should be an instruction to be followed therefor it should say HOW to do that action such as 'use a bash command to list all the files in X' or 'filter the files by running Y'.")

class MakePlanTool(LLMTool):
    def __init__(self):
        super().__init__(
            name="make_plan",
            description="Makes a step by step plan for the system to follow.",
            parameters=MakePlanToolParameters,
            requiredParameters=["steps"],
            type="function"
        )

class CanDoStepToolParameters(BaseModel):
    can_do_step: bool = Field(..., description="A boolean value indicating if the system can complete the step.")
    how: str = Field(..., description="A description of how to complete the step or a reason it cannot be completed. This must be descriptive.")

class CanDoStepTool(LLMTool):
    def __init__(self):
        super().__init__(
            name="can_do_step",
            description="Determine if the system can complete the step with the tools available.",
            parameters=CanDoStepToolParameters,
            requiredParameters=["can_do_step", "how"],
            type="function"
       )

#TODO Implement this
class MakePythonToolToolParameters(BaseModel):
    wanted_changes: str = Field(..., description="The changes to be made to the current plan and the reasons for making the changes.")

class UpdatePlanTool(LLMTool):
    def __init__(self):
        super().__init__(
            name="update_plan",
            description="Make changes to the current plan the system is following.",
            parameters=MakePythonToolToolParameters,
            requiredParameters=["wanted_changes"],
            type="function"
        )
