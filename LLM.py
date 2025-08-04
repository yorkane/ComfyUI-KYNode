import io
import os
import openai
import base64
import numpy as np
import requests
import json
from PIL import Image

_CATEGORY = "KYNode/LLM"
PROTOCOLS = ["openai", "ollama"]
MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "chatgpt-4o-latest",
    "gpt-4-turbo",
    "OpenGVLab/InternVL3-8B",
    "MiniCPM-V-2_6_awq"
]
OLLAMA_MODELS = [
    "aha2025/llama-joycaption-beta-one-hf-llava:Q8_0",
    "qwen2.5vl:7b",
    "qwen2.5vl:32b",
    "XiaomiMiMo/MiMo-VL-7B-RL",
    "openbmb/minicpm-o2.6",
    "openbmb/minicpm-v2.6"
]


class OpenAICaptionImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in": ("IMAGE", {}),
                "protocol": (PROTOCOLS, {"default": "openai"}),
                "custom_model": ("STRING", {"default": ""}),
                "model": (MODELS,),
                "ollama_model": (OLLAMA_MODELS,),
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
        protocol,
        custom_model,
        model,
        ollama_model,
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
        
        # 根据协议选择模型
        if protocol == "ollama":
            if not custom_model:
                custom_model = ollama_model
            return self._caption_ollama(
                pil_image, custom_model, system_prompt, caption_prompt, 
                max_tokens, temperature, base_url
            )
        else:  # openai
            if not custom_model:
                custom_model = model
            return self._caption_openai(
                pil_image, custom_model, system_prompt, caption_prompt,
                max_tokens, temperature, top_p, frequency_penalty, 
                presence_penalty, base_url, api_key
            )

    def _caption_openai(self, pil_image, model, system_prompt, caption_prompt,
                       max_tokens, temperature, top_p, frequency_penalty,
                       presence_penalty, base_url, api_key):
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Set up OpenAI client
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # Make API call to OpenAI
        response = client.chat.completions.create(
            model=model,
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
            timeout=30,
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

    def _caption_ollama(self, pil_image, model, system_prompt, caption_prompt,
                       max_tokens, temperature, base_url):
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Prepare Ollama API request
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": caption_prompt,
                    "images": [img_str]
                }
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if "message" in result and "content" in result["message"]:
                caption = result["message"]["content"].strip()
                return (caption,)
            else:
                raise ValueError("Invalid response format from Ollama")
        except Exception as e:
            raise ValueError(f"Ollama API error: {str(e)}")


class OpenAICaptionImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "protocol": (PROTOCOLS, {"default": "openai"}),
                "custom_model": ("STRING", {"default": ""}),
                "model": (MODELS,),
                "ollama_model": (OLLAMA_MODELS,),
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
    RETURN_NAMES = ("captions",)
    OUTPUT_IS_LIST = (True,)
    CATEGORY = _CATEGORY
    FUNCTION = "caption_batch"
    DESCRIPTION = "Caption multiple images using openai-protocol local LLM services (support MiniCPM-V)"

    def caption_batch(
        self,
        images,
        protocol,
        custom_model,
        model,
        ollama_model,
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
        # 根据协议选择模型
        if protocol == "ollama":
            if not custom_model:
                custom_model = ollama_model
            return self._caption_batch_ollama(
                images, custom_model, system_prompt, caption_prompt,
                max_tokens, temperature, base_url
            )
        else:  # openai
            if not custom_model:
                custom_model = model
            return self._caption_batch_openai(
                images, custom_model, system_prompt, caption_prompt,
                max_tokens, temperature, top_p, frequency_penalty,
                presence_penalty, base_url, api_key
            )

    def _caption_batch_openai(self, images, model, system_prompt, caption_prompt,
                             max_tokens, temperature, top_p, frequency_penalty,
                             presence_penalty, base_url, api_key):
        # 设置 OpenAI 客户端
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        # 处理输入图像
        if len(images.shape) == 4:  # 批次输入 (B,H,W,C)
            batch_size = images.shape[0]
            image_list = [images[i:i+1] for i in range(batch_size)]
        else:  # 单张图片输入
            image_list = [images]
        
        captions = []
        for img in image_list:
            try:
                # 转换图像格式
                pil_image = Image.fromarray(
                    np.clip(255.0 * img.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                )
                
                # 转换为 base64
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # 准备消息内容
                messages = [
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
                ]

                # 发送请求
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=30,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                
                if response.choices[0].message.content is None:
                    captions.append("Error: No content in response")
                else:
                    caption = response.choices[0].message.content.strip()
                    captions.append(caption)
                    
            except Exception as e:
                captions.append(f"Error: {str(e)}")
                
        return (captions,)

    def _caption_batch_ollama(self, images, model, system_prompt, caption_prompt,
                           max_tokens, temperature, base_url):
        # 处理输入图像
        if len(images.shape) == 4:  # 批次输入 (B,H,W,C)
            batch_size = images.shape[0]
            image_list = [images[i:i+1] for i in range(batch_size)]
        else:  # 单张图片输入
            image_list = [images]
        
        captions = []
        for img in image_list:
            try:
                # 转换图像格式
                pil_image = Image.fromarray(
                    np.clip(255.0 * img.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                )
                
                # 转换为 base64
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # Prepare Ollama API request
                url = f"{base_url}/api/chat"
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": caption_prompt,
                            "images": [img_str]
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }

                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                if "message" in result and "content" in result["message"]:
                    caption = result["message"]["content"].strip()
                    captions.append(caption)
                else:
                    captions.append("Error: Invalid response format from Ollama")
                    
            except Exception as e:
                captions.append(f"Error: {str(e)}")
                
        return (captions,)


class OpenAIChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "protocol": (PROTOCOLS, {"default": "openai"}),
                "custom_model": ("STRING", {"default": ""}),
                "model": (MODELS,),
                "ollama_model": (OLLAMA_MODELS,),
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
        protocol,
        custom_model,
        model,
        ollama_model,
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
        # 根据协议选择模型
        if protocol == "ollama":
            if not custom_model:
                custom_model = ollama_model
            return self._chat_ollama(
                custom_model, system_prompt, chat_prompt,
                max_tokens, temperature, base_url
            )
        else:  # openai
            if not custom_model:
                custom_model = model
            return self._chat_openai(
                custom_model, system_prompt, chat_prompt,
                max_tokens, temperature, top_p, frequency_penalty,
                presence_penalty, base_url, api_key
            )

    def _chat_openai(self, model, system_prompt, chat_prompt,
                    max_tokens, temperature, top_p, frequency_penalty,
                    presence_penalty, base_url, api_key):
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # Make API call to OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat_prompt},
            ],
            timeout=30,
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

    def _chat_ollama(self, model, system_prompt, chat_prompt,
                    max_tokens, temperature, base_url):
        # Prepare Ollama API request
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if "message" in result and "content" in result["message"]:
                result_text = result["message"]["content"].strip()
                return (result_text,)
            else:
                raise ValueError("Invalid response format from Ollama")
        except Exception as e:
            raise ValueError(f"Ollama API error: {str(e)}")


LLM_CLASS_MAPPINGS = {
    "KY_OpenAICaptionImage": OpenAICaptionImage,
    "KY_OpenAICaptionImages": OpenAICaptionImages,
    "KY_OpenAIChat": OpenAIChat,
}

LLM_NAME_MAPPINGS = {
    "KY_OpenAICaptionImage": "KY Caption Image by openai-protocol local LLM services",
    "KY_OpenAICaptionImages": "KY Caption Images Batch by openai-protocol local LLM services",
    "KY_OpenAIChat": "KY Chat with openai-protocol local LLM services",
}
