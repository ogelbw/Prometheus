from typing import List, Any, Dict, Tuple
from dotenv import load_dotenv
import pydantic
from os import getenv as env
import openai
import json

def get_tool_desc():
    class Test_tool(pydantic.BaseModel):
        some_str_parameter: str
        some_int_parameter: int = pydantic.Field(..., description="Should always be the number 56")
        some_list_parameter: list = pydantic.Field(..., description="A list of things in the format: [{\"action\" : str, \"reason\" : str}...].")
    return Test_tool

def test_function(arguments):
    """A simple test function, supplly any arguments."""
    # Do something with the arguments
    return arguments


if __name__ == "__main__":
    load_dotenv()
    client=openai.OpenAI(api_key=env("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model='gpt-4o-mini-2024-07-18',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please call the 'test_tool' tool. this is a test."},
            {"role": "user", "content": "For the list parameter provide multiple items."}
        ],
        stream=False,
        tools=[
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "Processes input parameters",
                "parameters": get_tool_desc().model_json_schema(),
            },
        }
    ],
        tool_choice={"function":{"name": "test_function"}, "type": "function"}
    )

    if response.choices[0].finish_reason == "stop":
        function_call = response.choices[0].message.tool_calls[0]
        if function_call.function.name == "test_function":
            # Parse the arguments using the Pydantic model
            arguments = json.loads(function_call.function.arguments)
            # arguments = function_call.function.arguments
            # Call your function with the validated arguments
            print(arguments)

            for entry in arguments['some_list_parameter']:
                print(entry, type(entry))
            # Use the result as needed


