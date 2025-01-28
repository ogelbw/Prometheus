import os
import platform
import getpass
import shutil
from datetime import datetime
import psutil  # For additional system information; requires the 'psutil' package

if __name__ != "__main__":
    # when called directly this import doesn't work. The tool description 
    # cannot be testing in this state
    from prometheus.tools.definitions import LLMTool, LLMToolParameter

def Run():
    # OS information
    os_name = platform.system() + " " + platform.release()
    
    # Shell information
    shell = os.environ.get("SHELL", "Unknown")
    
    # Desktop environment (for Linux, if available)
    desktop_env = os.environ.get("DESKTOP_SESSION", "Unknown")

    # Current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Storage information (total, used, free in GB)
    total, used, free = shutil.disk_usage("/")
    storage_info = {
        "total_gb": total // (2**30),
        "used_gb": used // (2**30),
        "free_gb": free // (2**30)
    }
    
    # Current user
    current_user = getpass.getuser()

    # System architecture
    architecture = platform.architecture()[0]
    
    # CPU information
    cpu_info = {
        "cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "frequency_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown"
    }

    # Memory information
    memory_info = {
        "total_gb": psutil.virtual_memory().total // (2**30),
        "available_gb": psutil.virtual_memory().available // (2**30)
    }

    # Python version
    python_version = platform.python_version()

    # Compile information into a dictionary
    os_info = {
        "os_name": os_name,
        "shell": shell,
        "desktop_env": desktop_env,
        "current_time": current_time,
        "storage_info": storage_info,
        "current_user": current_user,
        "architecture": architecture,
        "cpu_info": cpu_info,
        "memory_info": memory_info,
        "python_version": python_version
    }

    return os_info

def ToolDescription():
    return LLMTool(
        name="get_os_info",
        description="Get information about the OS.",
        parameters=[],
        requiredParameters=[],
        type="function"
    )

if __name__ == "__main__":
    os_info = Run()
    print(os_info)
