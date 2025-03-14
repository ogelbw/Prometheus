This is my third year undergrad project, The goal of this project is to make a framework that allows LLMs to carry out generic tasks by having them create a plan and create their own python tools in order to achieve that plan.

### Setup
1. install python 3.10 or higher
2. Set up a virtual environment (optional)
3. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
4. Run the demo script with the following command:
```bash
python run_prometheus_demo.py
```

### Scripting
The prometheus class can either be ran end to end with `Task(str)` where the agent will attempt to complete the task by calling it's own tools and creating new ones when needed.

```python
agent.Task("Combine all the files in the downloads folder into a single file") # Will run until the task is completed
```

 or you can control the execution of the agent by first calling `MakePlan` and then ExecuteStep() to run the agent step by step.
```python
# this will be set to false when the the agent calls 'task_complete'
agent.running_task = True 

# The agent will generate a plan.
agent.MakePlan("Combine all the files in the downloads folder into a single file") 

# then you can run the agent step by step
while agent.running_task:
    agent.ExecuteStep()
else:
    print("Task Complete")
```

#### Prompting
You can override the default prompts the agent uses. Each of the prompts are in the form of a list of dicts that follow the openai chat format. 

This lets you create specific prompts to beter fit your usecase. Prompts directly change how the LLMs behave and therefore bad prompts may stop te system from working correctly.

See `prometheus/default_prompts.py` For how the prompts are structured.

#### Creating a tool directly
You can also create a tool directly by calling the `CreatePythonTool` method and passing a dict in the openai chat format. There are functions for making these messages in `prometheus/tools/definitions.py`
```python
from prometheus.tools.definitions import User_msg

agent.CreatePythonTool(User_msg("Create a tool that combines all the files in the downloads folder into a single file"))
```

This will start the dev-reviewer dialogue and generate the specified python tool. The tool will then be saved to `tools_path` and can be used in future tasks. 

Note this will import the needed python modules to the currently active python environment and import the tool into the environment.

#### Porting other LLM services

In order for the prometheus class to use an LLM service it need to return it's generations/inferences as a `openai.types.chat.chat_completion.ChatCompletion` object for normal messages or a `openai.types.chat.chat_completion_message_tool_call.ChatCompletionMessageToolCall` object for tool calls.

The prometheus class takes 3 `llm_client_base` objects for the Executor, Dev and Reviewer. These objects are how the prometheus class interacts with LLM services.

The `llm_client_base` class has 2 methods that need to be implemented:
    - base_invoke: which is used to invoke the LLM service
    - GetModels : which returns a list of models available in the LLM service.

See `prometheus/utils.py` for how `llm_client_base` was extended for the `llm_client_openai` class.