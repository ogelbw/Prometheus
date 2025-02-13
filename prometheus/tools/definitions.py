import dataclasses
from typing import List, Literal, Callable

@dataclasses.dataclass
class LLMToolParameter:
    name: str
    type: Literal["string", "array", "number", "boolean", "integer", "object", "enum"]
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