import io
import os
import openai
import base64
import numpy as np
from PIL import Image

_CATEGORY = "KYNode/LLM"
MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "chatgpt-4o-latest",
    "gpt-4-turbo",
    "OpenGVLab/InternVL2_5-4B",
    "OpenGVLab/InternVL2_5-8B",
    "MiniCPM-V-2_6_awq",
    "MiniCPM-V-2_6",
    "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "Qwen/Qwen2.5-7B-Instruct",
    "./models/Qwen2.5-Coder-7B-Instruct",
]


class OpenAICaptionImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in": ("IMAGE", {}),
                "custom_model": ("STRING", {"default": ""}),
                "model": (MODELS,),
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": "You are a movie scene director"},
                ),
                "caption_prompt": (
                    "STRING",
                    {"default": "Describe this image without any speculations"},
                ),
                "max_tokens": ("INT", {"default": 200}),
                "temperature": ("FLOAT", {"default": 0.5}),
                "top_p": ("FLOAT", {"default": 0.9}),
                "frequency_penalty": ("FLOAT", {"default": 0.0}),
                "presence_penalty": ("FLOAT", {"default": 0.0}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:23333"}),
                "api_key": ("STRING", {"default": "sk-0123456"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_out",)
    CATEGORY = _CATEGORY
    FUNCTION = "caption"

    def caption(
        self,
        image_in,
        custom_model,
        model,
        system_prompt,
        caption_prompt,
        max_tokens,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
        base_url,
        api_key,
    ):
        # image to base64, image is bwhc tensor
        # Convert tensor to PIL Image
        pil_image = Image.fromarray(
            np.clip(255.0 * image_in.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
        if not custom_model:
            custom_model = model

        # Convert PIL Image to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Set up OpenAI client
        # api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # Make API call to OpenAI
        response = client.chat.completions.create(
            model=custom_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_str}"},
                        },
                    ],
                },
            ],
            timeout=3,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        if response.choices[0].message.content is None:
            raise ValueError("No content in response")

        # Extract and return the caption
        caption = response.choices[0].message.content.strip()
        return (caption,)


class OpenAIChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_model": ("STRING", {"default": ""}),
                "model": (MODELS,),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are a professional translator for chinese and english",
                    },
                ),
                "chat_prompt": ("STRING", {"default": "Hi"}),
                "max_tokens": ("INT", {"default": 200}),
                "temperature": ("FLOAT", {"default": 0.5}),
                "top_p": ("FLOAT", {"default": 0.9}),
                "frequency_penalty": ("FLOAT", {"default": 0.0}),
                "presence_penalty": ("FLOAT", {"default": 0.0}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:23333"}),
                "api_key": ("STRING", {"default": "sk-0123456"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_out",)
    CATEGORY = _CATEGORY
    FUNCTION = "chat"

    def chat(
        self,
        custom_model,
        model,
        system_prompt,
        chat_prompt,
        max_tokens,
        temperature,
        base_url,
        api_key,
        frequency_penalty,
        top_p,
        presence_penalty,
    ):
        if not custom_model:
            custom_model = model

        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # Make API call to OpenAI
        response = client.chat.completions.create(
            model=custom_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat_prompt},
            ],
            timeout=3,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        if response.choices[0].message.content is None:
            raise ValueError("No content in response")

        # Extract and return the caption
        result = response.choices[0].message.content.strip()
        return (result,)


LLM_CLASS_MAPPINGS = {
    "KY_OpenAICaptionImage": OpenAICaptionImage,
    "KY_OpenAIChat": OpenAIChat,
}

LLM_NAME_MAPPINGS = {
    "KY_OpenAICaptionImage": "KY Caption Image by openai-protocol local LLM services",
    "KY_OpenAIChat": "KY Chat with openai-protocol local LLM services",
}
