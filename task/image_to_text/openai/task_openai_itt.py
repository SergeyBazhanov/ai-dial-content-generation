import base64
from pathlib import Path

from task._utils.constants import API_KEY, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.model_client import DialModelClient
from task._models.role import Role
from task.image_to_text.openai.message import ContentedMessage, TxtContent, ImgContent, ImgUrl


def start() -> None:
    project_root = Path(__file__).parent.parent.parent.parent
    image_path = project_root / "dialx-banner.png"

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Create DialModelClient with GPT-4o model
    # Available models: gpt-4o, gpt-4o-mini, claude-3-5-sonnet, gemini-1.5-pro, etc.
    model_name = "gpt-4o"
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name=model_name,
        api_key=API_KEY
    )

    # Approach 1: Using base64 encoded image
    print(f"\n{'='*50} Base64 Image Analysis {'='*50}")
    base64_data_url = f"data:image/png;base64,{base64_image}"
    
    message_base64 = ContentedMessage(
        role=Role.USER,
        content=[
            TxtContent(text="What do you see on this picture? Describe it in detail."),
            ImgContent(image_url=ImgUrl(url=base64_data_url))
        ]
    )
    
    response_base64 = client.get_completion(messages=[message_base64])
    print(f"\nModel Response (base64):\n{response_base64.content}")

if __name__ == "__main__":
    start()