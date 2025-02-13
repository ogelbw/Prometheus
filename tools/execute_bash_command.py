if __name__ != "__main__": from prometheus.tools.definitions import LLMTool, LLMToolParameter


import subprocess

def execute_bash_command(command):
    """
    Executes a bash command on the system.

    Parameters:
    - command (str): The bash command to execute.
    
    Returns:
    - output (str or None): Output of the executed command, if any. Otherwise, return None.
    - error (str or None): Error message in case of failure. Otherwise, return None.
    """
    try:
        # Run the command and get the output
        result = subprocess.run(command, capture_output=True, text=True, check=True, shell=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return None, str(e.stderr).strip()

def Run(command):
    """
    Wrapper function to run the execute_bash_command and handle results.

    Parameters:
    - command (str): The bash command to execute.
    
    Returns:
    - output (str or None): Output of the executed command, if any. Otherwise, return None.
    - error (str or None): Error message in case of failure. Otherwise, return None.
    """
    print("Running command:", command)
    result = execute_bash_command(command)
    if isinstance(result, tuple):
        output, error = result
    else:
        output = result
        error = None
    
    return output, error

if __name__ == "__main__":
    # Example usage
    command = 'if [ -d ~/Downloads ]; then echo "exists"; fi'
    output, error = Run(command)
    
    if output:
        print("Output:", output)
    elif error:
        print("Error:", error)

def ToolDescription():
    return LLMTool(
        name="execute_bash_command",
        description="Executes a bash command on the system.",
        parameters=[LLMToolParameter(name='command', type='string', description='The bash command to execute.')],
        requiredParameters=['command'],
        type="function"
    )
