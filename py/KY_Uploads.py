import os
import json
import requests
import torch
import base64
from PIL import Image
import numpy as np
from io import BytesIO

_CATEGORY = "KYNode/uploads"

class KY_Uploads:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upload_urls": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter upload URLs, one per line"}),
                "headers": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter headers, one per line (key: value)"}),
            },
            "optional": {
                "input_video": ("VIDEO",),
                "input_image": ("IMAGE",),
                "input_audio": ("AUDIO",),
                "input_json": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter JSON data"}),
                "local_file_path": ("STRING", {"multiline": False, "default": "", "placeholder": "Enter local file path"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response_json",)
    FUNCTION = "upload_files"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Upload files (Video, Image, Audio, JSON) or local files to specified URLs with custom headers"

    def upload_files(self, upload_urls, headers, input_video=None, input_image=None, input_audio=None, 
                     input_json="", local_file_path=""):
        # 解析上传URL列表
        urls = [url.strip() for url in upload_urls.split('\n') if url.strip()]
        if not urls:
            raise ValueError("No upload URLs provided")
        
        # 解析请求头
        header_dict = {}
        for line in headers.split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                header_dict[key.strip()] = value.strip()
            elif line:  # 处理没有冒号但非空的行
                # 可能是只有键没有值的情况
                header_dict[line] = ""
        
        # 确定要上传的文件源
        file_data = None
        file_name = "upload_file"
        
        if local_file_path:
            # 处理本地文件路径
            # 去掉可能存在的引号
            local_file_path = local_file_path.strip().strip('"').strip("'")
            if os.path.exists(local_file_path):
                # 从本地文件路径读取
                with open(local_file_path, 'rb') as f:
                    file_data = f.read()
                file_name = os.path.basename(local_file_path)
            else:
                raise ValueError(f"Local file not found: {local_file_path}")
        elif input_json:
            # JSON数据
            file_data = input_json.encode('utf-8')
            file_name = "data.json"
        elif input_image is not None:
            # 图像数据
            file_data = self._image_to_bytes(input_image)
            file_name = "image.png"
        elif input_audio is not None:
            # 音频数据 (简化处理，实际可能需要更复杂的处理)
            file_name = "audio.wav"
        elif input_video is not None:
            # 视频数据 (简化处理，实际可能需要更复杂的处理)
            file_name = "video.mp4"
        else:
            raise ValueError("No valid input provided for upload")
        
        # 向每个URL发送上传请求
        responses = []
        for url in urls:
            try:
                # 直接发送二进制数据，并通过查询参数传递文件名
                headers_with_content_type = header_dict.copy()
                headers_with_content_type['Content-Type'] = 'application/octet-stream'
                
                # 构造带文件名参数的URL（如果服务器支持）
                import urllib.parse
                if '?' in url:
                    url_with_params = f"{url}&filename={urllib.parse.quote(file_name)}"
                else:
                    url_with_params = f"{url}?filename={urllib.parse.quote(file_name)}"
                
                response = requests.post(url_with_params, headers=headers_with_content_type, data=file_data)
                
                # 尝试解析JSON响应，如果失败则返回文本
                try:
                    response_data = response.json()
                except:
                    response_data = {"text": response.text}
                    
                # 构造响应对象
                response_obj = {
                    "url": url,
                    "status_code": response.status_code,
                    "response": response_data
                }
                
                # 如果响应中包含文件URL，则提取它
                file_url = self._extract_file_url(response_data)
                if file_url:
                    response_obj["file_url"] = file_url
                    
                responses.append(response_obj)
            except Exception as e:
                responses.append({
                    "url": url,
                    "status_code": 0,
                    "error": str(e)
                })
        
        # 返回响应的JSON字符串
        return (json.dumps(responses, indent=2, ensure_ascii=False),)

    def _image_to_bytes(self, image_tensor):
        """将图像张量转换为字节数据"""
        # 处理图像批次的情况
        if image_tensor.dim() == 4:
            # 批次图像，取第一张
            image_tensor = image_tensor[0]
        
        # 将tensor转换为PIL图像
        i = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # 转换为字节
        buf = BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    
    def _is_json_response(self, response):
        """检查响应是否为JSON格式"""
        content_type = response.headers.get('content-type', '')
        return 'application/json' in content_type
    
    def _extract_file_url(self, response_data):
        """从响应数据中提取文件URL"""
        if isinstance(response_data, dict):
            # 常见的包含URL的字段名
            url_keys = ['url', 'file_url', 'download_url', 'location', 'link']
            for key in url_keys:
                if key in response_data and isinstance(response_data[key], str):
                    return response_data[key]
            
            # 递归检查嵌套字典
            for value in response_data.values():
                if isinstance(value, dict):
                    url = self._extract_file_url(value)
                    if url:
                        return url
        return None

# 注册节点
NODE_CLASS_MAPPINGS = {
    "KY_Uploads": KY_Uploads,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KY_Uploads": "KY Upload Files",
}