import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
import platform
import string
from server import PromptServer
from aiohttp import web
import comfy.utils

from .utils.image_convert import pil2tensor

_CATEGORY = "KYNode/files"

class FilePathAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "file.txt"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("parent_dir", "file_stem", "file_extension", "file_name", "full_path")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "从文件路径中提取上层目录、文件名（不含扩展名）、扩展名、文件名（含扩展名）和完整路径"

    def execute(self, file_path):
        # 去掉可能存在的双引号
        file_path = file_path.strip('"')
        path = Path(file_path)

        parent_dir = str(path.parent) + "/"
        file_stem = path.stem
        file_extension = path.suffix
        file_name = path.name
        full_path = str(path.absolute())

        return (parent_dir, file_stem, file_extension, file_name, full_path)


class FileSequenceAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "image_001_v02.png"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "INT")
    RETURN_NAMES = ("sequence_pattern1", "start_number1", "sequence_pattern2", "start_number2")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = """分析文件序列名称，返回两个最长的数字序列：
    - sequence_pattern1: 第一个序列模板（最长序列）
    - start_number1: 第一个序列起始号
    - sequence_pattern2: 第二个序列模板（次长序列）
    - start_number2: 第二个序列起始号
    支持格式：
    - image_001_v02.png -> image_###_v02.png (1) 和 image_001_v##.png (2)
    - shot05_take003.jpg -> shot##_take###.jpg (5, 3)
    注意：#的数量与原始数字的位数完全一致
    """

    def execute(self, file_path):
        path = Path(file_path.strip('"'))
        file_name = path.stem  # 不含扩展名的文件名
        extension = path.suffix  # 扩展名
        
        # 查找文件名中的所有数字序列
        # 匹配模式：文件名中的连续数字（包括前导零）
        number_matches = list(re.finditer(r'(0*\d+)', file_name))
        
        # 按序列长度排序
        number_matches.sort(key=lambda x: len(x.group(1)), reverse=True)
        
        # 准备返回值
        patterns = []
        numbers = []
        
        # 处理找到的序列（最多取前两个最长的）
        for match in number_matches[:2]:
            number_str = match.group(1)
            # 保持前导零，使用原始字符串长度
            padding = len(number_str)
            # 转换为数字（去掉前导零）
            start_number = int(number_str)
            
            # 构建当前序列的模式
            temp_name = (
                file_name[:match.start(1)] + 
                '#' * padding + 
                file_name[match.end(1):]
            )
            
            patterns.append(f"{temp_name}{extension}")
            numbers.append(start_number)
        
        # 确保始终返回两组值，不足的用空值补充
        while len(patterns) < 2:
            patterns.append(f"{file_name}{extension}")
            numbers.append(0)
            
        return (patterns[0], numbers[0], patterns[1], numbers[1])




class KY_GetPath:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "prefix": ("STRING", {"default": "", "multiline": False, "placeholder": "Prefix (e.g. C:/Data/)"}),
                "suffix": ("STRING", {"default": "", "multiline": False, "placeholder": "Suffix (e.g. /output)"}),
                "create_missing": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False", "tooltip": "Force create the directory if it does not exist."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("full_path", "current_dir", "parent_dir", "filename")
    FUNCTION = "process_path"
    CATEGORY = "KY_Nodes/Path"

    def process_path(self, path, prefix="", suffix="", create_missing=False, unique_id=None):
        # 1. 基础路径处理
        raw_path = path
        
        # 2. 拼接前缀后缀 (使用 os.path.join 自动适配系统分隔符)
        if prefix:
            raw_path = os.path.join(prefix, raw_path)
        if suffix:
            raw_path = os.path.join(raw_path, suffix)

        # 3. 规范化路径 (处理混合斜杠、.. 等)
        # os.path.normpath 会根据当前操作系统将 / 转换为 \ (Windows) 或保持 / (Linux)
        full_path = os.path.normpath(raw_path)
        
        # 4. 获取绝对路径
        abs_path = os.path.abspath(full_path)

        # 5. 强制创建目录逻辑
        if create_missing:
            try:
                if not os.path.exists(abs_path):
                    # 递归创建目录
                    os.makedirs(abs_path, exist_ok=True)
                    print(f"[KY_GetPath] Created missing directory: {abs_path}")
            except Exception as e:
                print(f"[KY_GetPath] Failed to create directory {abs_path}. Error: {e}")

        # 6. 提取信息
        # 注意：如果刚刚执行了创建目录，os.path.isdir(abs_path) 现在将返回 True
        if os.path.isdir(abs_path):
            current_dir = abs_path
            filename = "" 
            parent_dir = os.path.dirname(abs_path)
        else:
            # 如果路径存在但不是目录（是文件），或者路径仍然不存在（create_missing=False 或 创建失败）
            current_dir = os.path.dirname(abs_path)
            filename = os.path.basename(abs_path)
            parent_dir = os.path.dirname(current_dir)

        return (abs_path, current_dir, parent_dir, filename)

# --- 工具函数 ---

def get_available_drives():
    """获取 Windows 所有可用盘符"""
    drives = []
    if platform.system() == "Windows":
        # 遍历 A-Z
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                drives.append({
                    "name": f"Drive {letter}:",
                    "type": "drive", 
                    "path": drive
                })
    return drives

# --- API 路由注册 ---

def register_routes():
    try:
        @PromptServer.instance.routes.post("/ky_utils/browse")
        async def browse_filesystem(request):
            data = await request.json()
            path = data.get("path", "")
            
            # 特殊标记：如果前端请求 "ROOT_DRIVES"，则返回盘符列表 (Windows专用)
            if path == "ROOT_DRIVES":
                return web.json_response({
                    "path": "My Computer",
                    "parent_path": "", # 顶层无法再向上
                    "files": get_available_drives()
                })

            # 默认路径处理
            if not path:
                path = os.getcwd()
            
            # 路径存在性检查
            if not os.path.exists(path):
                # 尝试修复：如果是文件路径被删了，尝试取父目录
                parent_fallback = os.path.dirname(path)
                if os.path.exists(parent_fallback):
                    path = parent_fallback
                else:
                    return web.json_response({"error": "Path does not exist", "path": path, "files": []})

            # 如果是文件，取其父目录进行浏览
            if not os.path.isdir(path):
                 path = os.path.dirname(path)

            # 规范化路径显示
            path = os.path.abspath(path)
            
            files = []
            folders = []
            parent_path = os.path.dirname(path)

            # --- Windows 特殊处理：检测是否到达根目录 ---
            is_windows = platform.system() == "Windows"
            is_root = False
            
            if is_windows:
                # 在 Windows 下，如果 dirname(path) == path，说明是 C:\ 这种根目录
                if os.path.splitdrive(path)[1] == os.sep or path == parent_path:
                    is_root = True
                    # 根目录的父级设为特殊标记，用于触发盘符列表
                    parent_path = "ROOT_DRIVES" 
            else:
                # Linux/Mac
                if path == "/":
                    is_root = True
                    parent_path = "" # Linux 根目录没有父级

            # --- 读取文件列表 ---
            try:
                # 只有非根目录(Linux)或非特殊情况才添加 ".." 
                # 但前端通常依赖 parent_path 按钮，这里为了列表完整性可以保留，
                # 或者由前端根据 parent_path 渲染返回按钮
                if parent_path:
                     folders.append({"name": "..", "type": "dir", "path": parent_path})

                for item in os.listdir(path):
                    if item.startswith('.'): continue 
                    
                    full_item_path = os.path.join(path, item)
                    
                    # 检查类型和权限
                    try:
                        if os.path.isdir(full_item_path):
                            folders.append({"name": item, "type": "dir", "path": full_item_path})
                        else:
                            files.append({"name": item, "type": "file", "path": full_item_path})
                    except OSError:
                        # 忽略无权访问的文件/链接
                        continue
                
                # 排序
                folders.sort(key=lambda x: x['name'].lower())
                files.sort(key=lambda x: x['name'].lower())
                
                result = folders + files
                return web.json_response({
                    "path": path, 
                    "parent_path": parent_path, # 关键：让后端告诉前端父级在哪里
                    "files": result
                })
                
            except PermissionError:
                return web.json_response({"error": "Permission Denied", "path": path, "files": []})
            except Exception as e:
                return web.json_response({"error": str(e), "path": path, "files": []})
                
    except Exception as e:
        print(f"KY_GetPath: Error registering routes: {e}")

register_routes()

NODE_CLASS_MAPPINGS = {
    "KY_GetPath": KY_GetPath,
    "KY_FilePathAnalyzer-": FilePathAnalyzer,
    "KY_FileSequenceAnalyzer": FileSequenceAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KY_FilePathAnalyzer-": "FilePath Analyzer",
    "KY_FileSequenceAnalyzer": "File Sequence Analyzer",
    "KY_GetPath": "KY File/Folder Path",
}
