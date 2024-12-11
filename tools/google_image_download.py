import requests
import os
from dotenv import load_dotenv
from random import randint

if __name__ != "__main__":
    from prometheus import LLMTool, LLMToolParameter

def Run(query, file_path):
    """
    Searches for an image using Google Custom Search API and downloads the first result to the specified file path.

    Parameters:
        query (str): The search term to query for.
        file_path (str): The local path to save the downloaded image (e.g., 'downloads/image.jpg').
        api_key (str): Your Google Custom Search API key.
        cse_id (str): Your Custom Search Engine (CSE) ID.

    Environment Variables:
        GOOGLE_API_KEY (str): Your Google Custom Search API key.
        CSE_KEY (str): Your Custom Search Engine (CSE) ID.

    Returns:
        bool: True if the image was successfully downloaded, False otherwise.
    """

    # TODO remove this
    # return f"Image successfully downloaded to /home/ogelbw/Documents/Prometheus/red_panda_image.jpg"

    # Retrieve API key and CSE ID from environment variables
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("CSE_KEY")

    if file_path == '':
        file_path = 'downloads/image.jpg'

    if not api_key or not cse_id:
        print("API key or CSE ID not found in environment variables.")
        return False

    # Google Custom Search API URL
    search_url = "https://www.googleapis.com/customsearch/v1"
    
    # Parameters for the search
    params = {
        "q": query,          # Search query
        "cx": cse_id,        # Custom Search Engine ID
        "key": api_key,      # Google API key
        "searchType": "image", # Search only for images
        "num": 1,            # Number of results (we only need the first image)
    }
    
    # Make the request to the API
    response = requests.get(search_url, params=params)
    response.raise_for_status()  # Raise an error for bad HTTP status codes
    
    # Parse the JSON response to get the image URL
    results = response.json()
    image_url = results["items"][0]["link"]

    # Set headers to include a User-Agent to avoid 403 error
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Fetch the image content
    image_response = requests.get(image_url, headers=headers)
    image_response.raise_for_status()  # Raise an error if image download fails
    
    # Ensure the directory exists
    # print(f"{file_path = }")
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Write the image content to the file path
    with open(file_path, "wb") as file:
        file.write(image_response.content)

    return f"Image successfully downloaded to {os.path.abspath(file_path)}"
    
def ToolDescription():
    return LLMTool(
        name="google_image_download",
        description="Searches for and downloads an image to a relative file path.",
        parameters=[
            LLMToolParameter(
                name="query",
                description="The search query for the image.",
                type="str"
            ),
            LLMToolParameter(
                name="file_path",
                description="The local path to save the downloaded image.",
                type="str"
            ),
        ],
        requiredParameters=["query", "file_path"],
        type="function"
    )

if __name__ == "__main__":
    Run("red panda", "Pictures/rp.jpg")
