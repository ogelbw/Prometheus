import subprocess
import sys

from prometheus import LLMTool, LLMToolParameter

def install_package(package_name):
    """Installs a package in the currently active Python environment."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Package '{package_name}' installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to install package '{package_name}'.")

def ToolDescription():
    return LLMTool(
        name="pip",
        description="Installs a package in the currently active Python environment.",
        parameters=["package_name"],
        requiredParameters=["package_name"],
        type="function"
    )

def Run(package_name:str):
    install_package(package_name)

if __name__ == "__main__":
    install_package("numpy") # installing numpy as a test