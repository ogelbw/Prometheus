if __name__ != "__main__": from prometheus.tools.definitions import LLMTool, LLMToolParameter


import subprocess
from typing import Optional

def open_image_with_viewer(image_path: str, viewer_app: str, viewer_app_path: Optional[str] = None) -> None:
    """
    Open an image file with a specified viewer application.
    
    Parameters:
        image_path (str): Path to the downloaded image file.
        viewer_app (str): Name of the viewer application.
        viewer_app_path (Optional[str]): Custom path to the viewer application. 
                                         Optional and defaults to None.

    Returns:
        None
    """
    
    def get_viewer_command(viewer_app: str, viewer_app_path: Optional[str]) -> Optional[str]:
        if not viewer_app or not viewer_app.strip():
            print("No viewer app provided.")
            return None
        
        if viewer_app.lower() in ["gwenview", "xlookphoto"]:
            command = [viewer_app.lower()] if not viewer_app_path else [viewer_app, viewer_app_path]
            command += [image_path]
        elif viewer_app.lower().startswith("custom:"):
            custom_path, _, image_name = viewer_app.partition(":")
            viewer_command = f"'{custom_path.strip()}'" if custom_path else "'not_a_viewer'"
            image_command = f"'{image_path}'"
            command = [viewer_command, image_command]
        else:
            print(f"Unknown viewer application: {viewer_app}")
            return None
        
        return " ".join(command)
    
    viewer_command = get_viewer_command(viewer_app, viewer_app_path)
    
    if viewer_command:
        try:
            subprocess.run(viewer_command, shell=True)
        except FileNotFoundError as e:
            print(f"Error opening image with the specified application: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

def Run(image_path: str, viewer_app: str, **kwargs) -> None:
    """
    Entry point for running the open_image_with_viewer function.
    
    Parameters:
        image_path (str): Path to the downloaded image file.
        viewer_app (str): Name of the viewer application.
        kwargs: Additional keyword arguments. Optional in this context but included for flexibility.
    """
    open_image_with_viewer(image_path, viewer_app, **kwargs)

if __name__ == "__main__":
    # Example usage
    Run("/home/ogelbw/Documents/Prometheus/red_panda_image.jpg", "gwenview")

def ToolDescription():
    return LLMTool(
        name="open_image_with_viewer",
        description="Open the downloaded image with a suitable viewer application.",
        parameters=[LLMToolParameter(name='image_path', type='str', description='Path to the downloaded image file'),
LLMToolParameter(name='viewer_app', type='str', description='Name of the viewer application (e.g., Gwenview, XLookPhoto)')],
        requiredParameters=['viewer_app', 'image_path'],
        type="function"
    )
