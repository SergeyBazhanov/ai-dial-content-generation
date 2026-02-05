import asyncio
from datetime import datetime

from task._models.custom_content import Attachment
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role

class Size:
    """
    The size of the generated image.
    """
    square: str = '1024x1024'
    height_rectangle: str = '1024x1792'
    width_rectangle: str = '1792x1024'


class Style:
    """
    The style of the generated image. Must be one of vivid or natural.
     - Vivid causes the model to lean towards generating hyper-real and dramatic images.
     - Natural causes the model to produce more natural, less hyper-real looking images.
    """
    natural: str = "natural"
    vivid: str = "vivid"


class Quality:
    """
    The quality of the image that will be generated.
     - ‘hd’ creates images with finer details and greater consistency across the image.
    """
    standard: str = "standard"
    hd: str = "hd"

async def _save_images(attachments: list[Attachment]):
    # Create DIAL bucket client
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:
        # Iterate through images from attachments
        for i, attachment in enumerate(attachments):
            if attachment.url:
                # Download image from bucket
                image_bytes = await bucket_client.get_file(attachment.url)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                extension = attachment.type.split('/')[-1] if attachment.type else 'png'
                filename = f"generated_image_{timestamp}_{i}.{extension}"
                
                # Save image locally
                with open(filename, "wb") as f:
                    f.write(image_bytes)
                
                print(f"Image saved: {filename}")


def start() -> None:
    # Create DialModelClient with DALL-E 3 model
    # For Google: use 'imagegeneration@005'
    model_name = "dall-e-3"
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name=model_name,
        api_key=API_KEY
    )
    
    # Create message with image generation prompt
    prompt = "Sunny day on Bali with beautiful beaches, palm trees, and crystal clear water"
    message = Message(
        role=Role.USER,
        content=prompt
    )
    
    # Configure image generation parameters via custom_fields
    custom_fields = {
        "size": Size.square,      # Options: '1024x1024', '1024x1792', '1792x1024'
        "quality": Quality.hd,     # Options: 'standard', 'hd'
        "style": Style.vivid       # Options: 'vivid', 'natural'
    }
    
    print(f"\nGenerating image for: '{prompt}'")
    print(f"   Size: {custom_fields['size']}")
    print(f"   Quality: {custom_fields['quality']}")
    print(f"   Style: {custom_fields['style']}")
    
    # Generate image
    response = client.get_completion(
        messages=[message],
        custom_fields=custom_fields
    )
    
    print(f"\nModel Response:\n{response.content}")
    
    # Get attachments from response and save images
    if response.custom_content and response.custom_content.attachments:
        print(f"\nDownloading {len(response.custom_content.attachments)} image(s)...")
        asyncio.run(_save_images(response.custom_content.attachments))
    else:
        print("\nNo images in response attachments")


if __name__ == "__main__":
    start()
