if __name__ != "__main__": from prometheus.tools.definitions import LLMTool, LLMToolParameter 
from pydantic import BaseModel, Field

import os
import platform
import argparse

def get_desktop_path():
    """Gets the path to the user's desktop based on the operating system."""
    if platform.system() == "Windows":
        return os.path.join(os.path.expanduser("~"), "Desktop")
    elif platform.system() == "Linux":
        return os.path.join(os.path.expanduser("~"), "Desktop")
    elif platform.system() == "Darwin":  # macOS
        return os.path.join(os.path.expanduser("~"), "Desktop")
    else:
        raise NotImplementedError("Unsupported operating system: " + platform.system())

def list_files_on_desktop(recursive=False, file_extension=None):
    """Returns a list of all files on the desktop. Includes files in subdirectories if recursive is True."""
    desktop_path = get_desktop_path()
    file_list = []

    try:
        if recursive:
            for dirpath, _, filenames in os.walk(desktop_path):
                for name in filenames:
                    if file_extension is None or name.endswith(file_extension):
                        file_list.append(os.path.join(dirpath, name))
        else:
            for name in os.listdir(desktop_path):
                if os.path.isfile(os.path.join(desktop_path, name)):
                    if file_extension is None or name.endswith(file_extension):
                        file_list.append(os.path.join(desktop_path, name))
    except FileNotFoundError:
        print("Desktop path not found.")
    except PermissionError:
        print("Permission denied to access the Desktop.")
    
    return file_list

def Run(recursive=False, file_extension=None):
    """Main function to execute the tool."""
    files = list_files_on_desktop(recursive, file_extension)
    return files

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='List files on the desktop.')
    parser.add_argument('--recursive', action='store_true', help='List files in subdirectories as well.')
    parser.add_argument('--extension', type=str, help='Filter files by specific extension (e.g., .txt)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    result = Run(recursive=args.recursive, file_extension=args.extension)

    # Output formatted results
    if result:
        print("Files on Desktop:")
        for idx, file in enumerate(result, start=1):
            print(f"{idx}: {file}")
    else:
        print("No files found.")

def ToolDescription():
    class tool_paramerters(BaseModel):
        recursive: bool = Field(..., description="Specify if the search should include files in subdirectories.")
        file_extension: str = Field(..., description="Specify a file extension to filter the results (e.g., .txt).")


        pass

    return LLMTool(
        name="list_files_on_desktop_tool",
        description="A tool to list all files on the user's desktop, with options for recursive searching and file type filtering.",
        parameters=tool_paramerters,
        requiredParameters=['recursive', 'file_extension'],
        type="function"
    )
