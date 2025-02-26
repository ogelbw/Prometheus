import subprocess
import sys
from typing import List

from pydantic import BaseModel

if __name__ != "__main__": from prometheus.tools.definitions import LLMTool, LLMToolParameter

def install_package(package_names):
    """Installs a package in the currently active Python environment."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *[package_name for package_name in package_names]])
        print(f"Package '{package_names}' installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to install package '{" ".join([package_name for package_name in package_names])}'.")

def ToolDescription():

    class tool_paramerters(BaseModel):
        package_names: List[str]

    return LLMTool(
        name="pip",
        description="Installs pip packages in the currently active Python environment.",
        parameters=tool_paramerters,
        requiredParameters=["package_names"],
        type="function"
    )

def Run(package_names:List[str]):
    install_package(package_names)

if __name__ == "__main__":
    install_package("numpy") # installing numpy as a test