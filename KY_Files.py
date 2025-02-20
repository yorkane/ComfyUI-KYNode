import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

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


FILE_CLASS_MAPPINGS = {
    "KY_FilePathAnalyzer-": FilePathAnalyzer,
    "KY_FileSequenceAnalyzer": FileSequenceAnalyzer,
}

FILE_NAME_MAPPINGS = {
    "KY_FilePathAnalyzer-": "FilePath Analyzer",
    "KY_FileSequenceAnalyzer": "File Sequence Analyzer",
}
