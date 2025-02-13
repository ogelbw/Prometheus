if __name__ != "__main__": from prometheus.tools.definitions import LLMTool, LLMToolParameter


import os
import platform

def list_desktop_files():
    # Get the current user's desktop path based on the OS
    if platform.system() == "Windows":
        desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
    elif platform.system() == "Linux":
        desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
    else:
        return "Unsupported operating system"

    try:
        # List all files in the desktop directory
        files = os.listdir(desktop_path)
        return "\n".join(files)
    except Exception as e:
        return str(e)

def Run():
    return list_desktop_files()

if __name__ == "__main__":
    print(Run())

def ToolDescription():
    return LLMTool(
        name="list_desktop_files",
        description="List all files on the desktop for the current user.",
        parameters=[],
        requiredParameters=[],
        type="function"
    )
