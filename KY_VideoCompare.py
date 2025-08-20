import torch
import numpy as np
from PIL import Image
import folder_paths
import os
import json

_CATEGORY = 'KYNode/video'

class VideoCompareNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["url", "video_file"], {
                    "default": "url"
                }),
            },
            "optional": {
                "video_a_url": ("STRING", {
                    "default": "https://sample-videos.com/zip/500kb/sample-video-30s.mp4",
                    "multiline": False
                }),
                "video_b_url": ("STRING", {
                    "default": "https://s3-us-west-2.amazonaws.com/s.cdpn.io/4273/floodplain-dirty.mp4?2222",
                    "multiline": False
                }),
                "video_a": ("VIDEO",),
                "video_b": ("VIDEO",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "compare_videos"
    CATEGORY = _CATEGORY
    DESCRIPTION = "对比两个视频（支持URL和视频文件）并在预览窗口中显示"
    OUTPUT_NODE = True

    def compare_videos(self, mode, video_a_url=None, video_b_url=None, video_a=None, video_b=None, unique_id=None, extra_pnginfo=None):
        """
        对比两个视频（支持URL和视频文件）
        
        Args:
            mode: 对比模式 ("url" 或 "video_file")
            video_a_url: 第一个视频URL
            video_b_url: 第二个视频URL
            video_a: 第一个视频文件对象
            video_b: 第二个视频文件对象
            
        Returns:
            包含视频信息的UI输出
        """
        video_a_source = ""
        video_b_source = ""
        
        if mode == "url":
            video_a_source = video_a_url if video_a_url else ""
            video_b_source = video_b_url if video_b_url else ""
        elif mode == "video_file":
            
            # 强制处理，即使对象为None也尝试处理
            video_a_source = self.process_video_object(video_a, "A") if video_a is not None else (video_a_url or "")
            print(f"Video A source: {video_a_source}")
            
            video_b_source = self.process_video_object(video_b, "B") if video_b is not None else (video_b_url or "")
            print(f"Video B source: {video_b_source}")
        
        # 确保返回的是字符串而不是数组
        if isinstance(video_a_source, list):
            video_a_source = ''.join(video_a_source)
        if isinstance(video_b_source, list):
            video_b_source = ''.join(video_b_source)

        
        # 返回UI信息，包含视频源
        result = {
            "ui": {
                "video_a_source": video_a_source,
                "video_b_source": video_b_source,
                "mode": mode
            },
            "result": ()
        }
        
        print(f"Returning result: {result}")
        return result
    
    def process_video_object(self, video_obj, label):
        """
        处理视频对象并生成预览URL
        
        Args:
            video_obj: 视频对象
            label: 标签（A或B）
            
        Returns:
            视频预览URL
        """
        
        if video_obj is None:
            print(f"Video {label} is None, returning empty string")
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
                print(f"Generated source for video {label} (VideoFromFile path): {source}")
                return source
            # 如果是BytesIO对象，我们无法直接获取文件名
            elif hasattr(file_attr, 'name'):
                # BytesIO可能有name属性
                filename = os.path.basename(getattr(file_attr, 'name', 'unknown.mp4'))
                source = f"/api/view?filename={filename}&type=input"
                print(f"Generated source for video {label} (VideoFromFile BytesIO): {source}")
                return source
            else:
                # 其他情况，转换为字符串
                source = str(file_attr)
                print(f"Generated source for video {label} (VideoFromFile str): {source}")
                return source
        
        # 处理其他可能的属性
        if hasattr(video_obj, 'filename'):
            filename = getattr(video_obj, 'filename', '')
            subfolder = getattr(video_obj, 'subfolder', '')
            type_ = getattr(video_obj, 'type', 'input')
            source = f"/api/view?type={type_}&filename={filename}&subfolder={subfolder}"
            return source
        
        if isinstance(video_obj, dict):
            print(f"Video {label} is dict: {video_obj}")
            # 检查常见的键
            if "filename" in video_obj:
                filename = video_obj["filename"]
                subfolder = video_obj.get("subfolder", "")
                type_ = video_obj.get("type", "input")
                source = f"/api/view?type={type_}&filename={filename}&subfolder={subfolder}"
                print(f"Generated source for video {label} (filename): {source}")
                return source
            elif "path" in video_obj:
                path = video_obj["path"]
                filename = os.path.basename(path)
                subfolder = os.path.dirname(path)
                source = f"/api/view?filename={filename}&subfolder={subfolder}&type=input"
                print(f"Generated source for video {label} (path): {source}")
                return source
            else:
                # 尝试其他可能的键
                keys = list(video_obj.keys())
                print(f"Video {label} dict keys: {keys}")
                source = str(video_obj)
                print(f"Generated source for video {label} (dict): {source}")
                return source
        elif isinstance(video_obj, str):
            print(f"Video {label} is string: {video_obj}")
            if os.path.exists(video_obj):
                filename = os.path.basename(video_obj)
                subfolder = os.path.dirname(video_obj)
                source = f"/api/view?filename={filename}&subfolder={subfolder}&type=input"
                print(f"Generated source for video {label} (file): {source}")
                return source
            else:
                source = video_obj
                print(f"Generated source for video {label} (url): {source}")
                return source
        else:
            # 其他类型，尝试转换为字符串并检查是否包含有用信息
            source = str(video_obj)
            print(f"Video {label} as string: {source}")
            
            # 如果字符串中包含filename信息，尝试提取
            if 'filename=' in source and 'type=' in source:
                return source
            else:
                return source

# 节点映射
VIDEO_COMPARE_CLASS_MAPPINGS = {
    "KY_VideoCompare": VideoCompareNode,
}

VIDEO_COMPARE_NAME_MAPPINGS = {
    "KY_VideoCompare": "Video Compare1111",
}