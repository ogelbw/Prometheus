import re
from prometheus.tools.definitions import LLMTool, MakeToolReviewerApproveTool, logging_codes
from typing import List, Dict, Callable, Tuple
from os import path
from logging import Logger
from prometheus.utils import llm_client_base, llm_client_interactions, llm_client_openai
from prometheus.tools.definitions import LLMTool, Tool_response, User_msg, System_msg, Assistant_msg
from copy import deepcopy


class Python_Tool_developer:
    def __init__(
        self,
        tools_path: str,
        step_retry_attempts: int,
        logger: Logger,
        llm_dev_client: llm_client_base,
        llm_reviewer_client: llm_client_base,
        developer_start_prompt: Callable,
        reviewer_start_prompt: Callable,
        pip_tool: LLMTool,
    ):
        self.llm_dev_client = llm_dev_client
        self.llm_reviewer_client = llm_reviewer_client
        self.tools_path = tools_path
        self.step_retry_attempts = step_retry_attempts
        self.log = logger
        self.developer_start_prompt = developer_start_prompt
        self.reviewer_start_prompt = reviewer_start_prompt
        self.llm_interactions = llm_client_interactions()
        self.pip_tool = pip_tool

        self.reviwer_tools = {
            "approve_tool": MakeToolReviewerApproveTool()
        }

        self.dev_tools = {}

    def install_required_modules(self, tool_name, pythonCode):
        """ " From the python code this Identifies the required python modules and installs them."""
        for i in range(self.step_retry_attempts):
            # get the tool to install all the needed packages
            installPackageReponse = self.llm_dev_client.base_invoke(
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an AI responsible for installing packages in the system.",
                    },
                    {
                        "role": "system",
                        "content": f"Install all the needed packages via the pip tool for the following python code: \n{pythonCode}",
                    },
                    {
                        "role": "system",
                        "content": f"Note the name for the python packages may differ from the import names, make sure to install the correct packages.",
                    }
                ],
                stream=False,
                use_tools=True,
                tools= {"pip": self.pip_tool},
                forced_tool="pip",
            )

            # see if the LLM tried to make use the pip tool
            try:
                packages: Dict[str, List[str]] = self.llm_interactions._getLLMToolCall(
                    self.llm_interactions._getLLMResponseTools(installPackageReponse)[0]
                )[1]
                idsToDel = []
                for package in packages["package_names"]:
                    if "prometheus" in package:
                        idsToDel.append(package)
                for id in idsToDel:
                    packages["package_names"].remove(id)

                # install the packages
                self.pip_tool.function(packages["package_names"])
                break
            except Exception as e:
                print(e)
                self.log(
                    f"Failed to install packages for tool: {tool_name}.",
                    level=self.ACTION_FAILED,
                )
                if i < self.step_retry_attempts - 1:
                    self.log(f"Trying again...", level=self.ACTION_FAILED)
                else:
                    raise RuntimeError(
                        f"Failed to install packages for tool: {tool_name}. No more attempts."
                    )

    # This method creates a new python tool for use by the system.
    # it will repeatedly create a python script, have the script reviewed by a LLM model where
    # it will point out potential issues with the current tool and suggestions on how to fix them,
    # the developer llm will then act on the suggestions. this repeats until the tool has been made.
    def MakeTool(
        self,
        tool_summary_msg: Dict[str, str],
        iteration_max: int = 20
    ):
        """Creates a python tool in the tools directory. returns the tool name."""
        running = True
        args = None

        developer_start_prompt = self.developer_start_prompt(
            use_developer=self.llm_dev_client.use_developer
        )
        reviewer_start_prompt = self.reviewer_start_prompt(
            use_developer=self.llm_reviewer_client.use_developer
        )

        # although it is not memory efficient to keep 2 copies of the chat it 
        # is faster than repeatedly filtering the messages to switch the roles.
        msg_history_dev = [tool_summary_msg]
        msg_history_reviewer = [tool_summary_msg]
        developer_speaking = True

        for i in range(iteration_max):
            response = self.llm_dev_client.base_invoke(
                stream=False,
                messages= (developer_start_prompt + msg_history_dev) if developer_speaking else (reviewer_start_prompt + msg_history_reviewer),
                use_tools=(i > 4),
                tools=self.dev_tools if developer_speaking else self.reviwer_tools,
            )

            if developer_speaking:
                self.log(response.choices[0].message.content or "No message", logging_codes.DEV_MSG.value,)
                msg_history_dev.append(
                    Assistant_msg(
                        msg=response.choices[0].message.content,
                        name="Developer",
                        tool_calls=response.choices[0].message.tool_calls
                    )
                )
                msg_history_reviewer.append(
                    User_msg(
                        msg=response.choices[0].message.content or "No message",
                        name="Developer",
                    )
                )
            else:
                self.log(response.choices[0].message.content or "No message", logging_codes.REVIEWER_MSG.value)
                msg_history_reviewer.append(
                    Assistant_msg(
                        msg=response.choices[0].message.content,
                        name="Reviewer",
                        tool_calls=response.choices[0].message.tool_calls
                    )
                )
                msg_history_dev.append(
                    User_msg(
                        msg=response.choices[0].message.content or "No message",
                        name="Reviewer",
                    )
                )

            # tool calling
            if response.choices[0].message.tool_calls:
                for toolCall in response.choices[0].message.tool_calls:
                    tool_name, args = self.llm_interactions._getLLMToolCall(toolCall)

                    if toolCall.function.name == "approve_tool":
                        # The Reviewer has deemed the tool ready for use, so stop.
                        running = False
                        break

                    # call the tool
                    tool_response = None
                    if developer_speaking:
                        tool_response = self.dev_tools[tool_name].function(**args)
                    else:
                        tool_response = self.reviwer_tools[tool_name].function(**args)
                    
                    # Adding the tool call response to the histories
                    if developer_speaking:
                        msg_history_dev.append(Tool_response(
                            call_id=toolCall.id,
                            content=tool_response
                        ))
                    else:
                        msg_history_reviewer.append(Tool_response(
                            call_id=toolCall.id,
                            content=tool_response
                        ))
                if not running: break
                # for-end, tool calling
            # if-end
            developer_speaking = not developer_speaking
        # for-end, development loop

        # Find the last developer msg that should have the python code
        for i in range (len(msg_history_dev) - 1, -1, -1):
            if msg_history_dev[i].get("name") == "Developer":
                break
        
        # get the python code from the message
        python_chunks = re.findall(r"```python\n(.*?)\n```", msg_history_dev[i].get("content"), re.DOTALL)
        largest_chunk = sorted(python_chunks, key=len, reverse=True)[0]

        # get the tool meta from the tool call
        Tool_name, description, parameters = args["Tool_name"], args["description"], args["parameters"]
        required_parameters = [parameter['name'] for parameter in parameters if parameter['default'] is not None]

        # Constructing the python code
        pythonCode = f"""if __name__ != "__main__": from prometheus.tools.definitions import LLMTool, LLMToolParameter \nfrom pydantic import BaseModel, Field

{largest_chunk}

def ToolDescription():
    class tool_paramerters(BaseModel):
{'\n'.join([f"        {parameter['name']}: {parameter['python_type']} = Field(..., description=\"{parameter['description']}\")" for parameter in parameters if parameter['name'] in required_parameters])}
{'\n'.join([f"        {parameter['name']}: Optional[{parameter['python_type']}] = Field(None, description=\"{parameter['description']}\")" for parameter in parameters if parameter['name'] not in required_parameters])}
        pass

    return LLMTool(
        name="{Tool_name}",
        description="{description}",
        parameters=tool_paramerters,
        requiredParameters={required_parameters},
        type="function"
    )
"""

        # Writing the python file to the tools directory
        with open(path.join(self.tools_path, f"{Tool_name}.py"), "w") as f:
            f.write(pythonCode)

        # self.install_required_modules(tool_name, pythonCode)
        return Tool_name
# End of MakeTool
