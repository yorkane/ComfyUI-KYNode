import torch
import numpy as np
from PIL import Image
import folder_paths
import os
import json

_CATEGORY = 'KYNode/video'


def process_video_object(video_obj):
    """
    处理视频对象并生成预览URL
    
    Args:
        video_obj: 视频对象
        
    Returns:
        视频预览URL
    """
    
    if video_obj is None:
        return ""
    
    # 处理ComfyUI的VideoFromFile对象
    if hasattr(video_obj, '_VideoFromFile__file'):
        file_attr = getattr(video_obj, '_VideoFromFile__file', None)            
        # 如果是字符串路径
        if isinstance(file_attr, str):
            filename = os.path.basename(file_attr)
            subfolder = os.path.dirname(file_attr)
            # 尝试确定文件类型
            if 'input' in file_attr:
                type_ = 'input'
            elif 'output' in file_attr:
                type_ = 'output'
            else:
                type_ = 'input'  # 默认为input
            source = f"/api/view?type={type_}&filename={filename}&subfolder={subfolder}"
            return source
        # 如果是BytesIO对象，我们无法直接获取文件名
        elif hasattr(file_attr, 'name'):
            # BytesIO可能有name属性
            filename = os.path.basename(getattr(file_attr, 'name', 'unknown.mp4'))
            source = f"/api/view?filename={filename}&type=input"
            return source
        else:
            # 其他情况，转换为字符串
            source = str(file_attr)
            return source
    
    # 处理其他可能的属性
    if hasattr(video_obj, 'filename'):
        filename = getattr(video_obj, 'filename', '')
        subfolder = getattr(video_obj, 'subfolder', '')
        type_ = getattr(video_obj, 'type', 'input')
        source = f"/api/view?type={type_}&filename={filename}&subfolder={subfolder}"
        return source
    
    if isinstance(video_obj, dict):
        # 检查常见的键
        if "filename" in video_obj:
            filename = video_obj["filename"]
            subfolder = video_obj.get("subfolder", "")
            type_ = video_obj.get("type", "input")
            source = f"/api/view?type={type_}&filename={filename}&subfolder={subfolder}"
            return source
        elif "path" in video_obj:
            path = video_obj["path"]
            filename = os.path.basename(path)
            subfolder = os.path.dirname(path)
            source = f"/api/view?filename={filename}&subfolder={subfolder}&type=input"
            return source
        else:
            # 尝试其他可能的键
            source = str(video_obj)
            return source
    elif isinstance(video_obj, str):
        if os.path.exists(video_obj):
            filename = os.path.basename(video_obj)
            subfolder = os.path.dirname(video_obj)
            source = f"/api/view?filename={filename}&subfolder={subfolder}&type=input"
            return source
        else:
            source = video_obj
            return source
    else:
        # 其他类型，尝试转换为字符串并检查是否包含有用信息
        source = str(video_obj)
        
        # 如果字符串中包含filename信息，尝试提取
        if 'filename=' in source and 'type=' in source:
            return source
        else:
            return source


class VideoCompareNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "video_a_url_or_filepath": ("STRING", {
                    "placeholder": "https://s3-us-west-2.amazonaws.com/s.cdpn.io/4273/floodplain-dirty.mp4",
                    "default": "",
                    "multiline": False
                }),
                "video_b_url_or_filepath": ("STRING", {
                    "placeholder": "https://s3-us-west-2.amazonaws.com/s.cdpn.io/4273/floodplain-dirty.mp4",
                    "default": "",
                    "multiline": False
                }),
                "video_a": ("VIDEO",),
                "video_b": ("VIDEO",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "compare_videos"
    CATEGORY = _CATEGORY
    DESCRIPTION = "对比两个视频（支持URL和视频文件）并在预览窗口中显示"
    OUTPUT_NODE = True

    def compare_videos(self, video_a_url_or_filepath="", video_b_url_or_filepath="", video_a=None, video_b=None):
        """
        对比两个视频（支持URL和视频文件）
        
        Args:
            video_a_url: 第一个视频URL或文件地址
            video_b_url: 第二个视频URL或文件地址
            video_a: 第一个视频文件对象
            video_b: 第二个视频文件对象
            
        Returns:
            包含视频信息的UI输出
        """
        video_a_source = ""
        video_b_source = ""
        
        if video_a is not None:
            video_a_source = process_video_object(video_a) if video_a is not None else (video_a_url_or_filepath or "")
        if video_b is not None:
            video_b_source = process_video_object(video_b) if video_b is not None else (video_b_url_or_filepath or "")
        
        # if video_a_url is empty
        
    
    
        if video_a_url_or_filepath is not None and not video_a_url_or_filepath.startswith("http"):
            video_a_url_or_filepath = process_video_object(video_a_url_or_filepath)
        if video_b_url_or_filepath is not None and not video_b_url_or_filepath.startswith("http"):
            video_b_url_or_filepath = process_video_object(video_b_url_or_filepath)
        video_a_source = video_a_url_or_filepath if video_a_url_or_filepath else video_a_source
        video_b_source = video_b_url_or_filepath if video_b_url_or_filepath else video_b_source
 
        # 确保返回的是字符串而不是数组
        if isinstance(video_a_source, list):
            video_a_source = ''.join(video_a_source)
        if isinstance(video_b_source, list):
            video_b_source = ''.join(video_b_source)

        
        # 返回UI信息，包含视频源
        result = {
            "ui": {
                "video_a_source": video_a_source,
                "video_b_source": video_b_source
            },
            "result": ()
        }
        
        print(f"Returning result: {result}")
        return result



class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")

class KY_ToVideoUrl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any1": (any_typ,),
            },
            "optional": {
                "any2": (any_typ, {"default": None}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("URL_1", "URL_2")
    FUNCTION = "convert_to_string"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将任意对象转换为包含视频文件地址的HTTP地址字符串"

    def convert_to_string(self, any1, any2=None):
        """
        将任意对象转换为包含视频文件地址的HTTP地址字符串
        
        Args:
            any1: 任意对象
            any2: 可选的第二个任意对象
            
        Returns:
            包含视频文件HTTP地址的字符串元组
        """
        def is_video_object(obj):
            """判断对象是否为视频对象"""
            if obj is None:
                return False
            # 检查是否为ComfyUI的VideoFromFile对象
            if hasattr(obj, '_VideoFromFile__file'):
                return True
            # 检查是否包含视频相关的属性
            if hasattr(obj, 'filename') or hasattr(obj, 'subfolder') or hasattr(obj, 'type'):
                return True
            # 检查是否为字典且包含视频相关键
            if isinstance(obj, dict) and ("filename" in obj or "path" in obj):
                return True
            return False

        def extract_video_url_from_string(s):
            """从字符串中尝试提取视频URL"""
            if not isinstance(s, str):
                s = str(s)
            
            # 使用正则表达式直接从字符串中提取第一个视频文件地址
            import re
            # 匹配常见的视频文件扩展名
            video_extensions = r'\.(mp4|avi|mov|mkv|webm|flv|wmv|mpeg|mpg|m4v|3gp|3g2|ogv|vob|mts|m2ts|ts)'
            # 查找第一个视频文件路径
            match = re.search(r"'([^']*"+video_extensions+r"[^']*)'", s, re.IGNORECASE)
            if not match:
                match = re.search(r'"([^"]*'+video_extensions+r'[^"]*)"', s, re.IGNORECASE)
            
            if match:
                video_path = match.group(1)
                # 如果是文件路径，转换为API视图URL
                if os.path.exists(video_path):
                    filename = os.path.basename(video_path)
                    # 获取相对于ComfyUI目录的子目录
                    try:
                        # 获取ComfyUI根目录
                        comfyui_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        # 计算相对路径
                        relative_path = os.path.relpath(video_path, comfyui_root)
                        subfolder = os.path.dirname(relative_path).replace('\\', '/')
                        # 从路径中确定type
                        path_parts = relative_path.split(os.sep)
                        if 'temp' in path_parts:
                            type_ = 'temp'
                        elif 'input' in path_parts:
                            type_ = 'input'
                        elif 'output' in path_parts:
                            type_ = 'output'
                        else:
                            type_ = 'input'  # 默认值
                    except:
                        # 如果计算相对路径失败，使用原来的逻辑
                        subfolder = os.path.dirname(video_path).replace('\\', '/')
                        type_ = 'input'
                    if subfolder == 'input' or subfolder == 'output' or subfolder == 'temp':
                        subfolder = ''
                    # 构建完整的URL参数
                    import urllib.parse
                    fullpath_encoded = urllib.parse.quote(video_path.replace('\\', '/'))
                    return f"/api/view?filename={filename}&subfolder={subfolder}&type={type_}&format=video%2Fh264-mp4&fullpath={fullpath_encoded}"
            
            # 如果已经是API视图URL，直接返回
            if '/api/view' in s and ('filename=' in s or 'type=' in s):
                return s
            
            # 尝试从JSON字符串中提取信息
            if s.startswith('{') and s.endswith('}'):
                try:
                    data = json.loads(s)
                    if isinstance(data, dict):
                        if "filename" in data:
                            filename = data["filename"]
                            subfolder = data.get("subfolder", "")
                            type_ = data.get("type", "input")
                            return f"/api/view?type={type_}&filename={filename}&subfolder={subfolder}"
                        elif "path" in data:
                            path = data["path"]
                            filename = os.path.basename(path)
                            subfolder = os.path.dirname(path)
                            return f"/api/view?filename={filename}&subfolder={subfolder}&type=input"
                except:
                    pass
            
            # 如果是文件路径，转换为API视图URL
            if os.path.exists(s):
                filename = os.path.basename(s)
                try:
                    # 获取ComfyUI根目录
                    comfyui_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    # 计算相对路径
                    relative_path = os.path.relpath(s, comfyui_root)
                    subfolder = os.path.dirname(relative_path).replace('\\', '/')
                    # 从路径中确定type
                    path_parts = relative_path.split(os.sep)
                    if 'temp' in path_parts:
                        type_ = 'temp'
                    elif 'input' in path_parts:
                        type_ = 'input'
                    elif 'output' in path_parts:
                        type_ = 'output'
                    else:
                        type_ = 'input'  # 默认值
                except:
                    # 如果计算相对路径失败，使用原来的逻辑
                    subfolder = os.path.dirname(s).replace('\\', '/')
                    type_ = 'input'
                
                # 构建完整的URL参数
                import urllib.parse
                fullpath_encoded = urllib.parse.quote(s.replace('\\', '/'))
                return f"/api/view?filename={filename}&subfolder={subfolder}&type={type_}&format=video%2Fh264-mp4&fullpath={fullpath_encoded}"
            
            # 否则返回原字符串
            return s

        # 处理any1
        if is_video_object(any1):
            result1 = process_video_object(any1)
        else:
            result1 = extract_video_url_from_string(any1)

        # 处理any2
        if any2 is None:
            result2 = ""
        elif is_video_object(any2):
            result2 = process_video_object(any2)
        else:
            result2 = extract_video_url_from_string(any2)

        return (result1, result2)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "KY_VideoCompare": VideoCompareNode,
    "KY_ToVideoUrl": KY_ToVideoUrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KY_VideoCompare": "Video Compare",
    "KY_ToVideoUrl": "Video Object To String",
}