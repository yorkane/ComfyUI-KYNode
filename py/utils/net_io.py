import io
import os
import json
import torch
import base64
import random
import requests
import re
from typing import List, Dict, Tuple
import folder_paths
import comfy.utils


from PIL import Image, ImageOps, ImageFilter
import numpy as np


def load_images_from_url(urls: List[str], keep_alpha_channel=False):
    images: List[Image.Image] = []
    masks: List[Image.Image] = []

    for url in urls:
        if url.startswith("data:image/"):
            i = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))
        elif url.startswith("http://") or url.startswith("https://"):
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                raise Exception(response.text)

            i = Image.open(io.BytesIO(response.content))
        elif url.startswith(("/view?", "/api/view?")):
            from urllib.parse import parse_qs

            qs_idx = url.find("?")
            qs = parse_qs(url[qs_idx + 1 :])
            filename = qs.get("name", qs.get("filename", None))
            if filename is None:
                raise Exception(f"Invalid url: {url}")

            filename = filename[0]
            subfolder = qs.get("subfolder", None)
            if subfolder is not None:
                filename = os.path.join(subfolder[0], filename)

            dirtype = qs.get("type", ["input"])
            if dirtype[0] == "input":
                url = os.path.join(folder_paths.get_input_directory(), filename)
            elif dirtype[0] == "output":
                url = os.path.join(folder_paths.get_output_directory(), filename)
            elif dirtype[0] == "temp":
                url = os.path.join(folder_paths.get_temp_directory(), filename)
            else:
                raise Exception(f"Invalid url: {url}")

            i = Image.open(url)
        elif url == "":
            continue
        # 使用正则表达式判断文件路径
        elif re.match(r'^file://[/\\]?([a-zA-Z]:[/\\]|/)', url):  # Windows: file:///C:/ 或 Linux: file:///
            # 移除file://前缀，保留路径部分
            if url.startswith('file:///'):
                url = url[8:]  # 移除file:///
            elif url.startswith('file://'):
                url = url[7:]  # 移除file://
            
            # Windows路径处理：将正斜杠转换为反斜杠
            if re.match(r'^[a-zA-Z]:', url):
                url = url.replace('/', '\\')
            
            if not os.path.isfile(url):
                 raise Exception(f"File {url} does not exist")
            i = Image.open(url)
         # 直接文件路径判断：Windows格式(C:\path) 或 Linux格式(/path)
        elif re.match(r'^([a-zA-Z]:[/\\]|/)', url):
             # 标准化Windows路径分隔符
             if re.match(r'^[a-zA-Z]:', url):
                 url = url.replace('/', '\\')
             
             if not os.path.isfile(url):
                 raise Exception(f"File {url} does not exist")
             i = Image.open(url)
        else:
             url = folder_paths.get_annotated_filepath(url)
             if not os.path.isfile(url):
                 raise Exception(f"Invalid url: {url}")

             i = Image.open(url)

        i = ImageOps.exif_transpose(i)
        has_alpha = "A" in i.getbands()
        mask = None

        if "RGB" not in i.mode:
            i = i.convert("RGBA") if has_alpha else i.convert("RGB")

        if has_alpha:
            mask = i.getchannel("A")

        if not keep_alpha_channel:
            image = i.convert("RGB")
        else:
            image = i

        images.append(image)
        masks.append(mask)

    return (images, masks)
