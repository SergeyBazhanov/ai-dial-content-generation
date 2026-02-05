import asyncio
from io import BytesIO
from pathlib import Path

from task._models.custom_content import Attachment, CustomContent
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role


async def _put_image() -> Attachment:
    file_name = 'dialx-banner.png'
    image_path = Path(__file__).parent.parent.parent / file_name
    mime_type_png = 'image/png'
    
    # Create DialBucketClient and upload image
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:
        # Open image file and load bytes
        with open(image_path, "rb") as image_file:
            image_bytes = BytesIO(image_file.read())
        
        # Upload file to bucket
        result = await bucket_client.put_file(
            name=file_name,
            mime_type=mime_type_png,
            content=image_bytes
        )
        
        # Return Attachment object with title, url and type
        return Attachment(
            title=file_name,
            url=result.get("url"),
            type=mime_type_png
        )


def start() -> None:
    # Create DialModelClient
    # Available models: gpt-4o, gpt-4o-mini, claude-3-5-sonnet, gemini-1.5-pro, etc.
    model_name = "gpt-4o"
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name=model_name,
        api_key=API_KEY
    )
    
    # Upload image using async function
    attachment = asyncio.run(_put_image())
    
    # Print attachment details
    print(f"\nUploaded Attachment:")
    print(f"   Title: {attachment.title}")
    print(f"   URL: {attachment.url}")
    print(f"   Type: {attachment.type}")
    
    # Create message with attachment
    message = Message(
        role=Role.USER,
        content="What do you see on this picture?",
        custom_content=CustomContent(attachments=[attachment])
    )
    
    # Call chat completion
    response = client.get_completion(messages=[message])
    print(f"\nModel Response:\n{response.content}")


if __name__ == "__main__":
    start()
