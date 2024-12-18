import io
import os
import openai
import base64
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "chatgpt-4o-latest",
    "gpt-4-turbo"
]

class OpenAICaptionImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in" : ("IMAGE", {}),
                "model": ("STRING", {"default": "gpt-4o"}),
                "system_prompt": ("STRING", {"default": "You are a movie scene director"}),
                "caption_prompt": ("STRING", {"default": "Describe this image"}),
                "max_tokens": ("INT", {"default": 300}),
                "temperature": ("FLOAT", {"default": 0.5}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:23333"}),
                "api_key": ("STRING", {"default": "sk-0123456"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_out",)
    CATEGORY = "openai"
    FUNCTION = "caption"

    def caption(self, image_in, model, system_prompt, caption_prompt, max_tokens, temperature, base_url, api_key):
        # image to base64, image is bwhc tensor

        # Convert tensor to PIL Image
        pil_image = Image.fromarray(np.clip(255. * image_in.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Set up OpenAI client
        #api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # Make API call to OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response.choices[0].message.content is None:
            raise ValueError("No content in response")

        # Extract and return the caption
        caption = response.choices[0].message.content.strip()
        return (caption,)
