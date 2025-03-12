if __name__ != "__main__": from prometheus.tools.definitions import LLMTool, LLMToolParameter 
from typing import List
from pydantic import BaseModel, Field


import os
import subprocess
import platform
from pydantic import BaseModel, Field

def open_image_with_viewer(image_files: str, viewer: str) -> str:
    # Determine the platform
    current_platform = platform.system()
    processes = []
    opened = 0

    for image in image_files:
        # Validate the image file path
        if not os.path.isfile(image):
            return f"Error: The file '{image}' does not exist."
        
        # Check the viewer
        if viewer.lower() != "gwenview":
            return f"Error: Unsupported viewer '{viewer}'. Only 'gwenview' is supported."
        
        try:
            if current_platform == "Linux":
                # Attempt to open the image with gwenview on Linux
                processes.append(subprocess.Popen([viewer, image]) ) # ensure gwenview is installed
                opened += 1
            elif current_platform == "Windows":
                # On Windows, we'll need to use start command to open the file
                processes.append(subprocess.Popen(["start", viewer, image], shell=True))
            else:
                return f"Error: Unsupported operating system '{current_platform}'."
        except Exception as e:
            return f"Error when attempting to open the image: {str(e)}"
    
    # Wait for all processes to finish
    for process in processes:
        process.wait()
    return f"Successfully opened {opened} images with {viewer}."

def Run(image_files: List[str], viewer: str = "gwenview") -> str:
    return open_image_with_viewer(image_files, viewer)

# To test the function, you can call Run(image_file, viewer)
# Example usage (this should be commented out or removed in actual implementation):
# if __name__ == "__main__":
#     print(Run("path/to/your/image.jpg"))

def ToolDescription():
    class tool_paramerters(BaseModel):
        image_files: List[str] = Field(..., description="The paths to the image files that needs to be opened.")
        viewer: str = Field(..., description="The name of the viewer application to be used, e.g., gwenview.")
        pass

    return LLMTool(
        name="open_image_with_viewer",
        description="Open an image file with the gwenview image viewer.",
        parameters=tool_paramerters,
        requiredParameters=['image_file', 'viewer'],
        type="function"
    )

if __name__ == "__main__":
    import os
    image_files = []
    viewer = "gwenview"

    # Get the image files from the current directory
    for file in os.listdir('/home/ogelbw/Pictures/sd/other/decorative/'):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_files.append(os.path.join('/home/ogelbw/Pictures/sd/other/decorative/', file))

    print(Run(image_files, viewer))