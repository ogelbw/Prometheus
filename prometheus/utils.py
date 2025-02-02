from typing import List, Any, Dict, override
from prometheus.tools.definitions import LLMTool
from openai import OpenAI
import json
import enum

class logging_codes(enum.Enum):
    TOOL_MADE = 50
    TOOL_USED = 51

    THINKING = 60
    ACTION_COMPLETE = 61
    ACTION_FAILED = 62

    PLAN_MADE = 70
    PLAN_FAILED = 71

    TASK_COMPLETE = 80


class llm_client_interactions:
    def __init__(self, step_retry_attempts:int = 3):
        self.step_retry_attempts = step_retry_attempts

    def _getLLMResponseTools(self, LLM_response) -> List[Any]:
        """ Returns the tools used in the response from the LLM
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


class llm_client_base:
    """ Base class for the LLM client interactions"""
    def __init__(self, retry_attempts:int = 3):
        self.retry_attempts = retry_attempts
        self.llm_interations = llm_client_interactions()

    def base_invoke(self,
                    model:str,
                    messages: List[Dict[str, str]],
                    stream: bool = False,
                    use_tools: bool = False,
                    forced_tool: str = None,
                    tools: Dict[str, LLMTool] = None):
        """ Returns the Invoke the LLM api with the given messages."""
        pass

    def force_function_call_invoke(self,
                               model:str,
                               messages: List[Dict[str, str]],
                               forced_tool: str,
                               tools: Dict[str, LLMTool],
                               stream: bool = False,):
        """ Tries to force the LLM to use a tool. to be called with a preformatted tool"""
        for i in range(self.retry_attempts):
            response = self.base_invoke(
                model=model,
                messages=messages,
                stream=stream,
                use_tools=True,
                forced_tool=forced_tool,
                tools=tools
            )

            if self.llm_interations._LLMResponseIsToolCall(response):
                response = self.llm_interations._getLLMToolCall(self.llm_interations._getLLMResponseTools(response)[0])[1]
                break
            else:
                print("The LLM response is not a tool call.")
                print(response)
                response = None
        return response

    def GetModels(self):
        """ Returns the models available"""
        pass

class llm_client_openai(llm_client_base):
    def __init__(self, openai:OpenAI, retry_attempts = 3):
        super().__init__(retry_attempts)
        self.openAI_client= openai
    
    @override
    def base_invoke(self,
                    model:str,
                    messages: List[Dict[str, str]],
                    stream: bool = False,
                    use_tools: bool = False,
                    forced_tool: str = None,
                    tools: Dict[str, LLMTool] = None):
        """ Returns the Invoke the LLM api with the given messages."""
        
        # see if a forced tool is being used
        if forced_tool and use_tools:
            if forced_tool not in tools:
                raise ValueError(f"The forced tool '{forced_tool}' is not in the tools dictionary.")
        
        return self.openAI_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            tools = self.llm_interations._getFormattedTools(tools) if use_tools else [],
            tool_choice = self.llm_interations._formatToolChoice(forced_tool) if forced_tool else None,
        )

    @override
    def GetModels(self):
        """ Returns the models available in the OpenAI API."""
        return [x["id"] for x in self.openAI_client.models.list().to_dict()["data"]]




# def _baseInvoke(self,
#                 messages: List[Dict[str, str]],
#                 stream: bool = False,
#                 use_tools: bool = False,
#                 forced_tool: str = None,
#                 tools: Dict[str, LLMTool] = None,
#                 ):
#     """ Returns the Invoke the LLM api with the given messages."""
#     tools = tools if tools else self.tools

#     if use_tools and forced_tool:
#         # check that the forced tool is in the tools list
#         if forced_tool not in tools:
#             raise ValueError("The forced tool is not in the tools list.", forced_tool, tools.keys())

#     response = self._client.chat.completions.create(
#         model=self.active_model,
#         messages=messages,
#         stream=stream,
#         tools = self._getFormattedTools(tools) if use_tools else [],
#         tool_choice = self._formatToolChoice(forced_tool) if forced_tool else None,
#     )
#     return response
