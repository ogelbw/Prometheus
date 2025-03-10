from typing import List, Any, Dict, override, Literal

import openai
from prometheus.tools.definitions import LLMTool, System_msg
from openai import OpenAI
import json
import enum
from logging import Logger
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

# TODO Replace all of the schema related thing with thhe pydantic equiverlent
# and use pydantic_function_tool to convert it to a json string schhema for the openai api
#  thhis is for consistancy and to make it easier to work with the tools

def filter_system_messages(messages: List[dict]):
    """Filters out the system messages from the messages."""
    return [x for x in messages if (x["role"] != "system" and x["role"] != "developer")]

class llm_client_interactions:
    def __init__(self, step_retry_attempts: int = 3):
        self.step_retry_attempts = step_retry_attempts

    def _getLLMResponseTools(self, LLM_response: ChatCompletion):
        """Returns the tools used in the response from the LLM
        (if it decided to use a tool)."""
        return LLM_response.choices[0].message.tool_calls

    def _getLLMToolCall(self, LLM_response_tool:ChatCompletionMessageToolCall):
        """Returns a tuple of the name of the tool called and it's arguments.
        (An item from _getLLMResponseTools)"""
        return (
            LLM_response_tool.function.name,
            json.loads(LLM_response_tool.function.arguments),
        )

    def _getFormattedTools(self, tools: Dict[str, LLMTool] = None):
        """Formats the tools dictionary into a list of tools that can be passed to
        the OpenAI API."""
        tmp = []

        for key, tool in tools.items():
            tmp.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters.model_json_schema(),
                    },
                }
            )

            # The api doesn't understand "data" type. I am using it as an indicator
            # for myself when we want a data in a json format. To do this we pretend
            # that a function exists that uses parameters which are the data we want.
            if tool.type == "data":
                tmp[-1]["type"] = "function"
        return tmp

    def _formatToolChoice(self, tool_choice: str):
        """Formats a name of a tool into a dictionary that can be passed to the
        OpenAI API."""
        return {"type": "function", "function": {"name": tool_choice}}

    def _getResponseStopReason(self, LLM_response: ChatCompletion):
        """Returns the reason the response from the LLM stopped."""
        return LLM_response.choices[0].finish_reason

    def _getLLMResponseMessage(self, LLM_response: ChatCompletion):
        """Returns the complete response from the LLM when the response is not
        streamed"""
        return LLM_response.choices[0].message

    def _LLMResponseIsToolCall(self, LLM_response: ChatCompletion):
        """Returns whether the response from the LLM is a tool call."""
        return self._getResponseStopReason(LLM_response) == "tool_calls"


class llm_client_base:
    """Base class for the LLM client interactions"""

    def __init__(self, retry_attempts: int = 3, model: str = "", logger: Logger = None):
        self.retry_attempts = retry_attempts
        self.llm_interations = llm_client_interactions()
        self.model = model
        self.logger = logger

    def base_invoke(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        use_tools: bool = False,
        forced_tool: str = None,
        tools: Dict[str, LLMTool] = None,
    ) -> ChatCompletion:
        """Returns the Invoke the LLM api with the given messages."""
        pass

    def force_function_call_invoke(
        self,
        messages: List[Dict[str, str]],
        forced_tool: str,
        tools: Dict[str, LLMTool],
        stream: bool = False,
    ):
        """Tries to force the LLM to use a tool. to be called with a preformatted tool"""
        for i in range(self.retry_attempts):
            response = self.base_invoke(
                messages=messages,
                stream=stream,
                use_tools=True,
                forced_tool=forced_tool,
                tools=tools,
            )

            if self.llm_interations._LLMResponseIsToolCall(response):
                response = self.llm_interations._getLLMToolCall(
                    self.llm_interations._getLLMResponseTools(response)[0]
                )[1]
                break
            else:
                if self.logger is not None:
                    self.logger.warning(
                        f"Failed to force the tool call. Retrying attempt {i+1}/{self.retry_attempts}"
                    )
                response = None
        return response

    def GetModels(self):
        """Returns the models available"""
        pass


class llm_client_openai(llm_client_base):
    def __init__(self,
                 openai: OpenAI,
                 use_developer_instead_of_system: bool = False,
                 retry_attempts=3,
                 model: str = "",
                 logger=None):
        super().__init__(retry_attempts=retry_attempts, model=model, logger=logger)
        self.use_developer = use_developer_instead_of_system
        self.openAI_client = openai

    @override
    def base_invoke(
        self,
        messages: List[dict],
        stream: bool = False,
        use_tools: bool = False,
        forced_tool: str = None,
        tools: Dict[str, LLMTool] = None,
        reasoning_effort: Literal["low", "medium", "high", None] = None,
    ):
        """Returns the Invoke the LLM api with the given messages."""
        if forced_tool and use_tools:
            if forced_tool not in tools:
                raise ValueError(
                    f"The forced tool '{forced_tool}' is not in the tools dictionary."
                )

        try:
            re = None
            msg_append = []
            if reasoning_effort and (self.model.__contains__("o1") or self.model.__contains__("o3-mini")):
                re = reasoning_effort

            elif reasoning_effort:
                # In the event we want to reason with a model that doesn't
                # support it, append a system/developer message to the messages
                # to fake the reasoning. this still leads to better results.
                msg_append = System_msg(
                    f"Prepend your message with a <thinking> tag and reason about your response. For example prepend your message with something like: <thinking> I think the best way to do this is to use a bash command to list all the files in X as I can run ls via bash. </thinking>",
                 )

            if re:
                llm_response = self.openAI_client.chat.completions.create(
                    model=self.model,
                    messages=messages + msg_append,
                    stream=stream,
                    tools=(
                        self.llm_interations._getFormattedTools(tools)
                        if use_tools
                        else None
                    ),
                    tool_choice=(
                        self.llm_interations._formatToolChoice(forced_tool)
                        if forced_tool
                        else None
                    ),
                    reasoning_effort=re
                )
            else:
                llm_response = self.openAI_client.chat.completions.create(
                    model=self.model,
                    messages=messages + msg_append,
                    stream=stream,
                    tools=(
                        self.llm_interations._getFormattedTools(tools)
                        if use_tools
                        else None
                    ),
                    tool_choice=(
                        self.llm_interations._formatToolChoice(forced_tool)
                        if forced_tool
                        else None
                    ),
                )

        except openai.RateLimitError as e:
            if e.code == "insufficient_quota":
                raise ValueError("The OpenAI API has ran out of credits.")
            else:
                raise RuntimeError(
                    "You've somehow hit the rate limit of the OpenAI API."
                )

        return llm_response

    @override
    def GetModels(self):
        """Returns the models available in the OpenAI API."""
        return [x["id"] for x in self.openAI_client.models.list().to_dict()["data"]]
