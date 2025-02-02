import re
from prometheus.tools.definitions import LLMTool, LLMToolParameter
from typing import List, Dict
from os import path
from logging import Logger

class Python_Tool_developer:
    def __init__(self,
                 tools_path: str,
                 step_retry_attempts: int,
                 logger:Logger,
                 dev_llm_model: str,
                 reviewer_llm_model: str
                 ):
        self.dev_llm_model = dev_llm_model
        self.reviewer_llm_model = dev_llm_model
        self.tools_path = tools_path
        self.step_retry_attempts = step_retry_attempts
        self.log = logger

    def MakeTool(self,
                tool_name: str,
                tool_description: str,
                tool_parameters: List[LLMToolParameter],
                tool_required_parameters: List[str],
                developer_comment: str
                ):
        """Creates a python tool in the tools directory."""
        # Create the tool prompt
        tool_prompt = self.prometheus.create_tool_prompt(tool_name, tool_description, tool_parameters, tool_required_parameters, developer_comment)
        response = self.prometheus.llm_interactions._baseInvoke(
            messages=tool_prompt,
            stream=False,
            use_tools=False
        )

        # Note the self._LLMResponseIsToolCall won't work here because
        # A 'function' is forced to be used. This is a thing with openAI's API
        pythonCode = ''
        completeResponse = self.prometheus.llm_interactions._getLLMResponseCompleteResponse(response)

        # the response is a complete chat response since it fails to
        # use reponse formatting. we can use regex to get the python code
        python_chunks = re.findall(r'```python\n(.*?)\n```', completeResponse, re.DOTALL)
        if python_chunks:
            for chunk in python_chunks:
                pythonCode += f"\n{chunk}"
                break # TODO see about removing this break, currently it takes only the first python code block as the LLM has a tendeciy to return multiple code blocks

        # prepend the imports and the ToolDescription function
        pythonCode = f"""if __name__ != "__main__": from prometheus.tools.definitions import LLMTool, LLMToolParameter

{pythonCode}

def ToolDescription():
    return LLMTool(
        name="{tool_name}",
        description="{tool_description}",
        parameters=[{",".join([f"LLMToolParameter(name='{parameter.name}', type='{parameter.type}', description='{parameter.description}')" for parameter in tool_parameters])}],
        requiredParameters={tool_required_parameters},
        type="function"
    )
"""

        # Writing the python file to the tools directory
        with open(path.join(self.tools_path, f"{tool_name}.py"), "w") as f:
            f.write(pythonCode)

        for i in range(self.step_retry_attempts):
            # get the tool to install all the needed packages
            installPackageReponse = self.prometheus.llm_interactions._baseInvoke(
                messages=[
                    {"role": "system", "content": f"You are an AI responsible for installing packages in the system."},
                    {"role": "system", "content": f"Install all the needed packages via the pip tool for the following python code."},
                    {"role": "system", "content": f"{pythonCode}"},
                ],
                stream=False,
                use_tools=True,
                tools={"pip": self.prometheus.tools["pip"]},
                forced_tool="pip"
            )

            # see if the LLM tried to make use the pip tool
            try:
                packages:Dict[str, List[str]] = self.prometheus.llm_interactions._getLLMToolCall(self.prometheus.llm_interactions._getLLMResponseTools(installPackageReponse)[0])[1]
                idsToDel = []
                for package in packages['package_names']:
                    if "prometheus" in package:
                        idsToDel.append(package)
                for id in idsToDel:
                    packages['package_names'].remove(id)
                self.prometheus._callTool("pip", packages)
                break
            except Exception as e:
                print(e)
                self.log(f"Failed to install packages for tool: {tool_name}.", level=self.ACTION_FAILED)
                if i < self.step_retry_attempts - 1:
                    self.log(f"Trying again...", level=self.ACTION_FAILED)
                else:
                    raise RuntimeError(f"Failed to install packages for tool: {tool_name}. No more attempts.")