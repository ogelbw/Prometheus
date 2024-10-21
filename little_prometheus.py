import json

from openai import OpenAI
from typing import Callable, List, Literal, Dict
import dataclasses

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

def formatTools(tools:Dict[str, LLMTool]):
    """ Formats the tools dictionary into a list of tools that can be passed to the OpenAI API."""
    tmp = []
    for key, value in tools.items():
        tmp.append({
            "type": "function",
            "function": {
            "name": value.name,
            "description": value.description,
            "parameters": {
                "type": "object",
                "properties": {x.name: {
                    "type": x.type, 
                    "description": x.description
                } for x in value.parameters},
                "required": value.requiredParameters,
            },
        }})

        # The api doesn't understand "data" type. I am using it as an indicator 
        # for myself when we want a data in a json format. To do this we pretend
        # that a function exists that uses parameters which are the data we want.
        if value.type == "data":
            tmp[-1]["type"] = "function"
    return tmp

def formatToolChoice(tool_choice:str):
    """ Formats a name of a tool into a dictionary that can be passed to the OpenAI API."""
    return  {"type": "function", "function": {"name": tool_choice}}

def getLLMResponseDelta(LLM_response):
    """ Returns the delta of the response from the LLM when the response is
    streamed"""
    return LLM_response.choices[0].delta.content

def getLLMResponseCompleteResponse(LLM_response):
    """ Returns the complete response from the LLM when the response is not
    streamed"""
    return LLM_response.choices[0].message.content

def getLLMResponseArgs(LLM_response):
    """ Returns the arguments of the response from the LLM."""
    return json.loads(LLM_response.choices[0].message.tool_calls[0].function.arguments)

def getLLMResponseTool(LLM_response):
    """ Returns the tool used in the response from the LLM."""
    return LLM_response.choices[0].message.tool_calls[0].function.name

def listModels(client:OpenAI):
    """ Returns the models available in the OpenAI API."""
    return [x["id"] for x in client.models.list().to_dict()["data"]]

if __name__ == "__main__":
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="None" # Required by the class but not used by ollama
    )
    models = listModels(client)

    # Creating the tools used by the LLM. In this case it for formatting data.
    tools:Dict[str, LLMTool] = {
        "get_city_population": LLMTool(
            name="get_city_population",
            description="Get the populations of a list of cities.",
            parameters=[
                LLMToolParameter(name="cities", 
                                 type="List[str]", 
                                 description="A list of cities.")],
            requiredParameters=["cities"],
            type="data"
        )
    }

    LLM_response = client.chat.completions.create(
        model=models[0],
        messages=[
            {"role": "system", "content": "You are a helpful assistant." + 
             "get the Populations of the capitals of the given countries."},
            {"role": "user", "content": "england, france and germany"},
        ],
        stream=False,
        tools = formatTools(tools),
        tool_choice= formatToolChoice(list(tools.keys())[0]),
    )

    args = getLLMResponseArgs(LLM_response)
    for city in args["cities"]:
        print(city)