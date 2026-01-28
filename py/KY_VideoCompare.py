import torch
import numpy as np
from PIL import Image
import folder_paths
import os
import json
import urllib.parse

_CATEGORY = 'KYNode/video'

def get_ky_url(path):
    if not path:
        return ""
    filename = os.path.basename(path)
    fullpath_encoded = urllib.parse.quote(path.replace('\\', '/'))
    return f"/ky_utils/stream/{filename}?path={fullpath_encoded}"

def resolve_path(filename, subfolder="", type_="input"):
    try:
        # handle special subfolder names if they come from the weird ComfyUI logic
        if subfolder == 'input' or subfolder == 'output' or subfolder == 'temp':
            subfolder = ""
            
        base_dir = folder_paths.get_directory_by_type(type_)
        if not base_dir:
            return ""
            
        # If subfolder is not empty, join it
        if subfolder:
            full_path = os.path.join(base_dir, subfolder, filename)
        else:
            full_path = os.path.join(base_dir, filename)
            
        return os.path.abspath(full_path)
    except Exception as e:
        print(f"Error resolving path: {e}")
        return ""


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
            # It might be a relative path or just a filename depending on how it was created
            # Usually ComfyUI internally stores relative path or filename for input directory
            # Let's try to resolve it.
            
            # Check if it has input/output/temp in path to guess type
            filename = os.path.basename(file_attr)
            subfolder = os.path.dirname(file_attr)
            
            if 'input' in file_attr:
                type_ = 'input'
            elif 'output' in file_attr:
                type_ = 'output'
            else:
                type_ = 'input'  # default

            # If absolute path, use it directly
            if os.path.isabs(file_attr) and os.path.exists(file_attr):
                return get_ky_url(file_attr)

            full_path = resolve_path(filename, subfolder, type_)
            if os.path.exists(full_path):
                return get_ky_url(full_path)
            
            # fallback if resolution failed but it might be a direct path
            return get_ky_url(file_attr)

        # 如果是BytesIO对象，我们无法直接获取文件名
        elif hasattr(file_attr, 'name'):
            # BytesIO可能有name属性
            # Warning: this might be a temporary file that doesn't exist on disk in a standard location
            filename = getattr(file_attr, 'name', 'unknown.mp4')
            if os.path.isabs(filename):
                return get_ky_url(filename)
            # Try to find it in input
            full_path = resolve_path(os.path.basename(filename), "", "input")
            return get_ky_url(full_path)
        else:
            # 其他情况，转换为字符串
            source = str(file_attr)
            if os.path.exists(source):
                return get_ky_url(source)
            return source
    
    # helper to extract fields
    def get_info(obj):
        fname = getattr(obj, 'filename', None) or obj.get('filename') if isinstance(obj, dict) else None
        sub = getattr(obj, 'subfolder', None) or obj.get('subfolder', "") if isinstance(obj, dict) else ""
        typ = getattr(obj, 'type', None) or obj.get('type', "input") if isinstance(obj, dict) else "input"
        path = getattr(obj, 'path', None) or obj.get('path', None) if isinstance(obj, dict) else None
        return fname, sub, typ, path

    fname, sub, typ, path = get_info(video_obj)

    if path:
        return get_ky_url(path)
    
    if fname:
        full_path = resolve_path(fname, sub, typ)
        return get_ky_url(full_path)

    if isinstance(video_obj, str):
        if os.path.exists(video_obj):
            return get_ky_url(video_obj)
        else:
            # check if it looks like a relative path in input
            full_path = resolve_path(os.path.basename(video_obj), os.path.dirname(video_obj), "input")
            if os.path.exists(full_path):
                 return get_ky_url(full_path)
            return video_obj

    # fallback
    source = str(video_obj)
    # check if it already has filename/type info from some string representation
    if 'filename=' in source and 'type=' in source:
        # It's an API View URL probably, we can't easily convert it back without parsing
        # But if the user wants to use ky_utils, we should try.
        # But for safety, maybe just return it if we can't parse?
        # Let's leave it for now.
        return source
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
                if os.path.exists(video_path):
                    return get_ky_url(video_path)
            
            # 如果已经是API视图URL，尝试转换 (Optional, but good for compatibility if input is already a view url)
            if '/ky_utils/stream' in s:
                return s
            
            # 尝试从JSON字符串中提取信息
            if s.startswith('{') and s.endswith('}'):
                try:
                    data = json.loads(s)
                    if isinstance(data, dict):
                        # try to get absolute path
                        fname = data.get("filename")
                        sub = data.get("subfolder", "")
                        typ = data.get("type", "input")
                        path = data.get("path")
                        
                        if path and os.path.exists(path):
                            return get_ky_url(path)
                        if fname:
                            full_path = resolve_path(fname, sub, typ)
                            if os.path.exists(full_path):
                                return get_ky_url(full_path)
                except:
                    pass
            
            # 如果是文件路径
            if os.path.exists(s):
                return get_ky_url(s)
            
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

def _ensure_output_subdir():
    base = folder_paths.get_output_directory()
    subfolder = "ky_compare"
    out_dir = os.path.join(base, subfolder)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, subfolder

def _to_uint8_rgb(arr):
    arr = np.array(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3):
            pass
        elif arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype in (np.float32, np.float64):
        m = float(np.nanmax(arr)) if arr.size else 1.0
        if m <= 1.0:
            arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr

def save_image_to_output(img, name_prefix="img"):
    out_dir, subfolder = _ensure_output_subdir()
    filename = f"{name_prefix}_{abs(hash(str(img)))}.png"
    fullpath = os.path.join(out_dir, filename)
    if isinstance(img, Image.Image):
        img.save(fullpath)
    elif isinstance(img, np.ndarray):
        Image.fromarray(_to_uint8_rgb(img)).save(fullpath)
    elif isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
        Image.fromarray(_to_uint8_rgb(arr)).save(fullpath)
    else:
        return ""
    return get_ky_url(fullpath)

def process_image_object(image_obj):
    if image_obj is None:
        return ""
    
    # helper to check dict
    fname = getattr(image_obj, 'filename', None) or image_obj.get('filename') if isinstance(image_obj, dict) else None
    sub = getattr(image_obj, 'subfolder', None) or image_obj.get('subfolder', "") if isinstance(image_obj, dict) else ""
    typ = getattr(image_obj, 'type', None) or image_obj.get('type', "input") if isinstance(image_obj, dict) else "input"
    path = getattr(image_obj, 'path', None) or image_obj.get('path', None) if isinstance(image_obj, dict) else None

    if path:
        return get_ky_url(path)

    if fname:
        full_path = resolve_path(fname, sub, typ)
        return get_ky_url(full_path)

    if isinstance(image_obj, dict):
        if "image" in image_obj:
            return save_image_to_output(image_obj["image"])
        if "images" in image_obj:
            imgs = image_obj["images"]
            if isinstance(imgs, (list, tuple)) and imgs:
                return save_image_to_output(imgs[0])
                
    if isinstance(image_obj, str):
        if os.path.exists(image_obj):
            return get_ky_url(image_obj)
        # Try resolving as input file
        full_path = resolve_path(os.path.basename(image_obj), os.path.dirname(image_obj), "input")
        if os.path.exists(full_path):
            return get_ky_url(full_path)
        return image_obj
        
    # save arbitrary object to output and return view url
    return save_image_to_output(image_obj)


class ImageCompareNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image_a_url_or_filepath": ("STRING", {
                    "placeholder": "http(s)://... or local path",
                    "default": "",
                    "multiline": False
                }),
                "image_b_url_or_filepath": ("STRING", {
                    "placeholder": "http(s)://... or local path",
                    "default": "",
                    "multiline": False
                }),
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "compare_images"
    CATEGORY = 'KYNode/image'
    DESCRIPTION = "对比两张图片（支持URL、本地文件或图像对象）"
    OUTPUT_NODE = True

    def compare_images(self, image_a_url_or_filepath="", image_b_url_or_filepath="", image_a=None, image_b=None):
        image_a_source = ""
        image_b_source = ""

        if image_a is not None:
            image_a_source = process_image_object(image_a)
        if image_b is not None:
            image_b_source = process_image_object(image_b)

        if image_a_url_or_filepath is not None and not str(image_a_url_or_filepath).startswith("http"):
            image_a_url_or_filepath = process_image_object(image_a_url_or_filepath)
        if image_b_url_or_filepath is not None and not str(image_b_url_or_filepath).startswith("http"):
            image_b_url_or_filepath = process_image_object(image_b_url_or_filepath)

        image_a_source = image_a_url_or_filepath if image_a_url_or_filepath else image_a_source
        image_b_source = image_b_url_or_filepath if image_b_url_or_filepath else image_b_source

        if isinstance(image_a_source, list):
            image_a_source = ''.join(image_a_source)
        if isinstance(image_b_source, list):
            image_b_source = ''.join(image_b_source)

        result = {
            "ui": {
                "image_a_source": image_a_source,
                "image_b_source": image_b_source
            },
            "result": ()
        }
        print(f"Returning image result: {result}")
        return result


NODE_CLASS_MAPPINGS["KY_ImageCompare"] = ImageCompareNode
NODE_DISPLAY_NAME_MAPPINGS["KY_ImageCompare"] = "Image Compare"
