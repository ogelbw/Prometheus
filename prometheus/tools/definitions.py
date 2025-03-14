import enum
import dataclasses
from typing import Dict, List, Literal, Callable, Type, Tuple, Optional
from pydantic import BaseModel, Field

class logging_codes(enum.Enum):
    TOOL_MADE = 50
    DEV_MSG = 51
    REVIEWER_MSG = 52
    TOOL_USED = 53
    TOOL_CREATION = 54

    THINKING = 60
    ACTION_COMPLETE = 61
    ACTION_FAILED = 62

    PLAN_MADE = 70
    PLAN_FAILED = 71

    TASK_COMPLETE = 80

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
class LLMTool:
    """ A tool that can be used by the LLM."""
    name: str
    description: str
    parameters: BaseModel
    requiredParameters: List[str]
    type: Literal["function", "data"]
    function: Callable|None = None

# --- Special LLM tool calls ---
class MakePythonToolToolParameters(BaseModel):
    # description: str = Field(..., description="A description of the tool to be made.")
    # tool_name: str = Field(..., description="The name of the tool to be made.")
    pass

class MakePythonToolTool(LLMTool):
    def __init__(self):
        super().__init__(
            name="make_tool",
            description="Create a new python tool for the LLM to use.",
            parameters=MakePythonToolToolParameters,
            # requiredParameters=["description", "tool_name"],
            requiredParameters=[],
            type="function"
        )

class TaskCompleteToolParameters(BaseModel):
    pass

class TaskCompleteTool(LLMTool):
    def __init__(self):
        super().__init__(
            name="task_complete",
            description="Report that the system has completed the user's task.",
            parameters=TaskCompleteToolParameters,
            requiredParameters=[],
            type="function"
        )

class parameter(BaseModel):
    name: str = Field(..., description="The name of the parameter.")
    python_type: Literal['str', 'float', 'int', 'bool', 'List[str]', 'List[float]', 'List[int]', 'List[bool]'
     ] = Field(..., description="The type of the parameter as str.")
    description: str = Field(..., description="A description of what the parameter does.")
    default: Optional[str] = Field(..., description="The default value of the parameter. If set this parameter is optional.")

class MakeToolReviewerApproveToolParameters(BaseModel):
    Tool_name: str = Field(..., description="The name of the tool that was made. This is also used as the filename so don't use spaces or special characters.")
    description: str = Field(..., description="A description of what the tool does.")
    parameters: List[parameter] = Field(..., description="The parameters that the tool takes.")

class MakeToolReviewerApproveTool(LLMTool):
    def __init__(self):
        super().__init__(
            name="approve_tool",
            description="Call this is approve the code made by the developer and send it to the user.",
            parameters=MakeToolReviewerApproveToolParameters,
            requiredParameters=["Tool_name", "description", "parameters"],
            type="function"
        )

class updatePlanParameters(BaseModel):
    new_plan: str = Field(..., description="A step by step plan for how you are going to achieve the user's task. You should mention the tools you are going to call, what you are going to do with the result of that call (if anything) and the reason for doing it. You should also mention what you are going to do if the tool call fails for each step of the plan.")

class updatePlan(LLMTool):
    def __init__(self):
        super().__init__(
            name="update_plan",
            description="Update the plan that is pinned as the most recent message in the chat.",
            parameters=updatePlanParameters,
            requiredParameters=["new_plan"],
            type="function"
        )