from typing import List

from prometheus.tools.definitions import (
    InstructionResponse,
    LLMToolParameter,
    instruction,
)


def _makeToolPromptDefault(
    tool_name: str,
    tool_description: str,
    tool_parameters: List[LLMToolParameter],
    tool_required_parameters: List[str],
    comment: str,
):
    """Default prompt to create a tool."""
    return [
        {
            "role": "system",
            "content": f"You are an experienced python AI programmer who has been tasked with creating a python script called {tool_name} with the description: {tool_description}, for part of a larger system.",
        },
        {
            "role": "system",
            "content": f"The tool must have the following parameters: {', '.join([f'{x.name} ({x.type}) : {x.description}' for x in tool_parameters])}.",
        },
        {
            "role": "system",
            "content": f"The following are required parameters: {', '.join(tool_required_parameters)}. The rest are optional.",
        },
        {"role": "system", "content": f"Create a single python script. "},
        {
            "role": "system",
            "content": f"The script must have a global function called 'Run' so it can be called by the system. 'Run' must call the function you make and it must return a string (even if it is empty). This is because stdout is ignored by the system.",
        },
        {
            "role": "system",
            "content": f"The python script must be runnable on linux and windows systems, it should detect which system it is on and act accordingly.",
        },
        {
            "role": "system",
            "content": f"Additionally this comment to the developer was left: {comment}",
        },
    ]


def _makePlanPromptDefault(final_goal: str, plan: List[instruction], tools: List[dict]):
    return [
        {
            "role": "system",
            "content": f"You are a AI planner part of a larger system. This system can make it's own tools in python and is already logged into the user's machine.",
        },
        {
            "role": "system",
            "content": f"The user gave the system this goal: <{final_goal}>.",
        },
        {
            "role": "system",
            "content": f"This is the current plan to achieve the goal:\n{[f"{i}. {x.action}, Reason: {x.reason}" for i, x in enumerate(plan)]}",
        },
        {
            "role": "system",
            "content": f"The system has access to these tools: [{', '.join(tools.keys())}]",
        },
        {"role": "system", "content": f"The system can create new tools in python."},
        {"role": "system", "content": f"Create a plan to achieve the user's goal."},
    ]


def _updatePlanPromptDefault(
    final_goal: str,
    plan: List[instruction],
    previous_action_response: InstructionResponse,
    tools: List[dict],
):
    return [
        {
            "role": "system",
            "content": f"You are a AI planner part of a larger system. This system can make it's own tools in python and is already logged into the user's machine.",
        },
        {
            "role": "system",
            "content": f"The user gave the system this goal: <{final_goal}>.",
        },
        {
            "role": "system",
            "content": f"This is the current plan to achieve the goal:\n{[f"{i}. {x.action}, Reason: {x.reason}" for i, x in enumerate(plan)]}",
        },
        {
            "role": "system",
            "content": f"The system has access to these tools: [{', '.join(tools.keys())}]",
        },
        {
            "role": "system",
            "content": f"The systems previous action was: {previous_action_response.action} with the result: {previous_action_response.response}.",
        },
        {
            "role": "system",
            "content": f"Update the plan (by making a new one) to capture the result of the previous action.",
        },
    ]


def _takeActionStepPromptDefault(
    step: instruction, how: str, executionHistory: List[InstructionResponse], goal: str
):
    return [
        {
            "role": "system",
            "content": f"You are an AI responsible for taking actions in a system.",
        },
        {
            "role": "system",
            "content": f"Here is the previous execution history of the system: {". ".join([x.response for x in executionHistory])}",
        },
        {"role": "system", "content": f"The user gave the system this goal: <{goal}>."},
        {
            "role": "system",
            "content": f"The current step is to: <{step.action}> because: <{step.reason}>.",
        },
        {
            "role": "system",
            "content": f"The system thinks this can be done by: <{how}>.",
        },
        {"role": "system", "content": f"Do this step as described."},
    ]
