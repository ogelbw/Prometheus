import os
import platform
import getpass
import shutil
from datetime import datetime

from prometheus import LLMTool, LLMToolParameter

def Run():
    # OS information
    os_name = platform.system() + " " + platform.release()
    
    # Shell information
    shell = os.environ.get("SHELL", "Unknown")
    
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
    
    # Compile information into a dictionary
    os_info = {
        "os_name": os_name,
        "shell": shell,
        "current_time": current_time,
        "storage_info": storage_info,
        "current_user": current_user
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