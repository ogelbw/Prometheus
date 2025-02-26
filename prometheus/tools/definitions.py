import dataclasses
from typing import Dict, List, Literal, Callable, Type
from pydantic import BaseModel, Field

# TODO Replace the LLMToolParameter with a pydantic model, this is cleaner and more consistent
# also this would mean that LLMTool wouldn't need to have a list of parameters as it is now a schema
@dataclasses.dataclass
class LLMToolParameter:
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
    steps: list = Field(..., description="The steps for the plan in the format: [{\"action\" : str, \"reason\" : str}...].")

class MakePlanTool(LLMTool):
    def __init__(self):
        super().__init__(
            name="make_plan",
            description="Makes a step by step plan for the system to follow.",
            parameters=MakePlanToolParameters(),
            requiredParameters=["steps"],
            type="function"
        )

class CanDoStepToolParameters(BaseModel):
    can_do_step: bool = Field(..., description="A boolean value indicating if the system can complete the step.")
    how: str = Field(..., description="A description of how to complete the step or a reason it cannot be completed.")

class CanDoStepTool(LLMTool):
    def __init__(self):
        super().__init__(
            name="can_do_step",
            description="Determine if the system can complete the step with the tools available.",
            parameters=CanDoStepToolParameters(),
            requiredParameters=["can_do_step", "how"],
            type="function"
       )

class MakePythonToolToolParameters(BaseModel):
    tool_name: str = Field(..., description="The self descriptive name of the python tool.")
    tool_description: str = Field(..., description="A description of the tool.") 
    tool_parameters: list = Field(..., description="The parameters of the tool in the format: [{\"name\" : str, \"type\" : Literal[\"str\", \"int\", \"list\", \"float\", \"bool\"], \"description\" : str}...].")
    tool_required_parameters: List[str] = Field(..., description="The required parameters for the tool")
    dev_comment: str = Field(..., description="A comment to the developer of the tool. Any additional requirements not captured by the description.")

class MakePythonToolTool(LLMTool):
    def __init__(self):
        super().__init__(
            name="make_tool",
            description="Makes a python tool for the system.",
            parameters=MakePythonToolToolParameters(),
            requiredParameters=["tool_name", "tool_description", "tool_parameters", "tool_required_parameters", "dev_comment"],
            type="function"
        )