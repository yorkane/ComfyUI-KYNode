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
import os
try:
    import folder_paths
except Exception:
    folder_paths = None

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




class KY_GetFromPath:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "create_missing_folder": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False", "tooltip": "Create the directory only if it does not exist AND path ends with '/' or '\\'."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("full_path", "current_dir", "parent_dir", "filename", "image", "text")
    FUNCTION = "process_path"
    CATEGORY = "KY_Nodes/Path"

    def process_path(self, path, create_missing_folder=False, unique_id=None):
        # 1. 基础路径处理
        raw_path = path

        # 2. 规范化路径 (处理混合斜杠、.. 等)
        # os.path.normpath 会根据当前操作系统将 / 转换为 \ (Windows) 或保持 / (Linux)
        full_path = os.path.normpath(raw_path)
        
        # 3. 获取绝对路径
        abs_path = os.path.abspath(full_path)

        # 4. 强制创建目录逻辑
        if create_missing_folder:
            try:
                if not os.path.exists(abs_path):
                    # 只有当路径以 '/' 或 '\' 结尾时才创建目录
                    if raw_path.endswith('/') or raw_path.endswith('\\') or raw_path.endswith(os.sep):
                        # 递归创建目录
                        os.makedirs(abs_path, exist_ok=True)
                        print(f"[KY_GetFromPath] Created missing directory: {abs_path}")
                    else:
                        print(f"[KY_GetFromPath] Path does not end with '/' or '\\', not creating directory: {abs_path}")
            except Exception as e:
                print(f"[KY_GetFromPath] Failed to create directory {abs_path}. Error: {e}")

        # 5. 提取信息
        # 注意：如果刚刚执行了创建目录，os.path.isdir(abs_path) 现在将返回 True
        if os.path.isdir(abs_path):
            current_dir = abs_path
            filename = "" 
            parent_dir = os.path.dirname(abs_path)
        else:
            # 如果路径存在但不是目录（是文件），或者路径仍然不存在（create_missing_folder=False 或 创建失败）
            current_dir = os.path.dirname(abs_path)
            filename = os.path.basename(abs_path)
            parent_dir = os.path.dirname(current_dir)

        # 6. 初始化返回对象
        image_tensor = None
        text_content = ""

        # 7. 检测文件类型并加载相应对象
        if os.path.isfile(abs_path):
            file_ext = Path(abs_path).suffix.lower()
            
            # 图像文件类型
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico', '.tiff', '.tif'}
            # 文本文件类型
            text_extensions = {'.txt', '.md', '.log', '.csv', '.tsv', '.json', '.xml'}
            
            try:
                if file_ext in image_extensions:
                    # 加载图像
                    from PIL import Image, ImageOps
                    img = Image.open(abs_path)
                    img = ImageOps.exif_transpose(img)
                    
                    if img.mode == 'I':
                        img = img.point(lambda i: i * (1 / 255))
                    img = img.convert('RGB')
                    
                    # 转换为ComfyUI的tensor格式
                    import numpy as np
                    image_np = np.array(img).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np)[None,]
                    
                elif file_ext in text_extensions:
                    # 读取文本内容
                    try:
                        with open(abs_path, 'r', encoding='utf-8') as f:
                            text_content = f.read()
                    except UnicodeDecodeError:
                        # 尝试其他编码
                        try:
                            with open(abs_path, 'r', encoding='gbk') as f:
                                text_content = f.read()
                        except:
                            text_content = f"无法读取文件内容: {abs_path}"
                    except Exception as e:
                        text_content = f"读取文件时出错: {str(e)}"
                        
            except Exception as e:
                print(f"[KY_GetFromPath] Error loading file {abs_path}: {e}")

        return (abs_path, current_dir, parent_dir, filename, image_tensor, text_content)

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
        @PromptServer.instance.routes.post("/ky_utils/check_path")
        async def check_path_type(request):
            """检查路径是文件还是目录"""
            data = await request.json()
            path = data.get("path", "")
            
            if not path:
                return web.json_response({"error": "No path provided"})
            
            # 规范化路径
            path = os.path.normpath(path.strip('"'))
            
            # 检查路径是否存在
            if not os.path.exists(path):
                return web.json_response({"error": "Path does not exist", "type": "none"})
            
            # 检查是文件还是目录
            if os.path.isfile(path):
                return web.json_response({"type": "file", "path": path})
            elif os.path.isdir(path):
                return web.json_response({"type": "directory", "path": path})
            else:
                return web.json_response({"type": "unknown", "path": path})
        
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
                default_output = None
                try:
                    if folder_paths is not None:
                        default_output = folder_paths.get_output_directory()
                except Exception:
                    default_output = None
                if not default_output:
                    default_output = os.path.join(os.getcwd(), "output")
                path = default_output if os.path.exists(default_output) else os.getcwd()
            
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

                # 定义需要过滤的文件扩展名和目录名
                filtered_extensions = {'.py', '.pyw', '.pyc', '.pyo', '.pyd',  # Python相关
                                      '.sh', '.bash', '.zsh', '.fish',         # Shell脚本
                                      '.bat', '.cmd', '.ps1', '.psm1', '.psd1', # Windows批处理和PowerShell
                                      '.ini', '.cfg', '.conf', '.config',       # 配置文件
                                      '.yaml', '.yml', '.toml'}                 # 其他配置文件格式
                
                filtered_dir_names = {'__pycache__', '.git', '.svn', '.hg',      # 版本控制和Python缓存
                                     'node_modules', '.vscode', '.idea'}        # 开发工具目录

                for item in os.listdir(path):
                    if item.startswith('.'): continue
                    
                    full_item_path = os.path.join(path, item)
                    
                    # 检查类型和权限
                    try:
                        if os.path.isdir(full_item_path):
                            # 过滤特定目录名
                            if item in filtered_dir_names:
                                continue
                            folders.append({"name": item, "type": "dir", "path": full_item_path})
                        else:
                            # 获取文件扩展名并检查是否需要过滤
                            _, ext = os.path.splitext(item.lower())
                            if ext in filtered_extensions:
                                continue
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
        print(f"KY_GetFromPath: Error registering routes: {e}")

    try:
        @PromptServer.instance.routes.post("/ky_utils/file_preview")
        async def file_preview(request):
            data = await request.json()
            path = data.get("path", "")
            if not path or not os.path.exists(path) or not os.path.isfile(path):
                return web.json_response({"error": "Invalid file"})
            
            # 获取文件扩展名并检查是否为过滤类型
            ext = Path(path).suffix.lower()
            filtered_extensions = {'.py', '.pyw', '.pyc', '.pyo', '.pyd',  # Python相关
                                  '.sh', '.bash', '.zsh', '.fish',         # Shell脚本
                                  '.bat', '.cmd', '.ps1', '.psm1', '.psd1', # Windows批处理和PowerShell
                                  '.ini', '.cfg', '.conf', '.config',       # 配置文件
                                  '.yaml', '.yml', '.toml'}                 # 其他配置文件格式
            
            # 如果是过滤的文件类型，返回错误
            if ext in filtered_extensions:
                return web.json_response({"error": "File type not supported for preview"})
            
            ext = ext.lstrip(".")
            size = 0
            try:
                size = os.path.getsize(path)
            except Exception:
                pass
            mtime = 0
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                pass
            image_ext = {"jpg","jpeg","png","gif","bmp","svg","webp","ico","tiff","tif"}
            text_ext = {"txt","md","rtf","log","csv","tsv","json","xml"}  # 移除了ini, cfg, conf, yaml, yml, toml
            audio_ext = {"mp3","wav","flac","aac","ogg","wma","m4a","opus"}
            video_ext = {"mp4","avi","mkv","mov","wmv","flv","webm","m4v","3gp","ogv"}
            ftype = "unknown"
            if ext in image_ext:
                ftype = "image"
            elif ext in text_ext:
                ftype = "text"
            elif ext in audio_ext:
                ftype = "audio"
            elif ext in video_ext:
                ftype = "video"
            can_preview = False
            preview_url = None
            snippet = None
            if ftype == "image":
                can_preview = True
                preview_url = f"/ky_utils/stream?path={path}"
            elif ftype == "text":
                can_preview = True
                try:
                    with open(path, "rb") as f:
                        data_bytes = f.read(8192)
                    snippet = data_bytes.decode("utf-8", errors="replace")
                except Exception:
                    snippet = ""
            elif ftype in ("audio","video"):
                can_preview = True
                preview_url = f"/ky_utils/stream?path={path}"
            return web.json_response({
                "type": ftype,
                "size": size,
                "modified": mtime,
                "can_preview": can_preview,
                "preview_url": preview_url,
                "snippet": snippet
            })

        @PromptServer.instance.routes.get("/ky_utils/stream")
        async def stream_file(request):
            q = request.rel_url.query
            path = q.get("path")
            if not path or not os.path.exists(path) or not os.path.isfile(path):
                return web.Response(status=404)
            
            # 获取文件扩展名并检查是否为过滤类型
            ext_with_dot = Path(path).suffix.lower()
            filtered_extensions = {'.py', '.pyw', '.pyc', '.pyo', '.pyd',  # Python相关
                                  '.sh', '.bash', '.zsh', '.fish',         # Shell脚本
                                  '.bat', '.cmd', '.ps1', '.psm1', '.psd1', # Windows批处理和PowerShell
                                  '.ini', '.cfg', '.conf', '.config',       # 配置文件
                                  '.yaml', '.yml', '.toml'}                 # 其他配置文件格式
            
            # 如果是过滤的文件类型，返回404
            if ext_with_dot in filtered_extensions:
                return web.Response(status=404)
            
            ext = ext_with_dot.lstrip(".")
            image_ext = {"jpg","jpeg","png","gif","bmp","svg","webp","ico","tiff","tif"}
            audio_ext = {"mp3","wav","flac","aac","ogg","wma","m4a","opus"}
            video_ext = {"mp4","avi","mkv","mov","wmv","flv","webm","m4v","3gp","ogv"}
            mime = "application/octet-stream"
            mime_map = {
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "png": "image/png",
                "gif": "image/gif",
                "bmp": "image/bmp",
                "svg": "image/svg+xml",
                "webp": "image/webp",
                "ico": "image/x-icon",
                "tiff": "image/tiff",
                "tif": "image/tiff",
                "mp3": "audio/mpeg",
                "wav": "audio/wav",
                "flac": "audio/flac",
                "aac": "audio/aac",
                "ogg": "audio/ogg",
                "wma": "audio/x-ms-wma",
                "m4a": "audio/mp4",
                "opus": "audio/opus",
                "mp4": "video/mp4",
                "avi": "video/x-msvideo",
                "mkv": "video/x-matroska",
                "mov": "video/quicktime",
                "wmv": "video/x-ms-wmv",
                "flv": "video/x-flv",
                "webm": "video/webm",
                "m4v": "video/x-m4v",
                "3gp": "video/3gpp",
                "ogv": "video/ogg",
            }
            if ext in mime_map:
                mime = mime_map[ext]
            try:
                if ext in image_ext:
                    with open(path, "rb") as f:
                        content = f.read()
                    return web.Response(body=content, content_type=mime)
                elif ext in audio_ext or ext in video_ext:
                    file_size = os.path.getsize(path)
                    range_header = request.headers.get("Range")
                    if range_header:
                        try:
                            units, rng = range_header.split("=", 1)
                            if units != "bytes":
                                return web.Response(status=416)
                            start_str, end_str = rng.split("-", 1)
                            start = int(start_str) if start_str else 0
                            end = int(end_str) if end_str else file_size - 1
                            if start >= file_size:
                                return web.Response(status=416)
                            if end >= file_size:
                                end = file_size - 1
                            length = end - start + 1
                            with open(path, "rb") as f:
                                f.seek(start)
                                data = f.read(length)
                            headers = {
                                "Content-Range": f"bytes {start}-{end}/{file_size}",
                                "Accept-Ranges": "bytes",
                                "Content-Length": str(len(data)),
                            }
                            return web.Response(status=206, body=data, headers=headers, content_type=mime)
                        except Exception:
                            return web.Response(status=500)
                    with open(path, "rb") as f:
                        content = f.read()
                    headers = {
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(len(content)),
                    }
                    return web.Response(body=content, headers=headers, content_type=mime)
                return web.Response(status=415)
            except Exception:
                return web.Response(status=500)
    except Exception as e:
        print(f"KY_GetFromPath: Error registering preview routes: {e}")

register_routes()

NODE_CLASS_MAPPINGS = {
    "KY_GetFromPath": KY_GetFromPath,
    "KY_FilePathAnalyzer-": FilePathAnalyzer,
    "KY_FileSequenceAnalyzer": FileSequenceAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KY_FilePathAnalyzer-": "FilePath Analyzer",
    "KY_FileSequenceAnalyzer": "File Sequence Analyzer",
    "KY_GetFromPath": "KY File/Folder Path",
}
