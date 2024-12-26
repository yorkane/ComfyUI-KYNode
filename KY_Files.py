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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("parent_dir", "file_stem", "file_extension", "full_path")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "从文件路径中提取上层目录、文件名（不含扩展名）、扩展名和完整路径"

    def execute(self, file_path):
        # 去掉可能存在的双引号
        file_path = file_path.strip('"')
        path = Path(file_path)

        parent_dir = str(path.parent) + "/"
        file_stem = path.stem
        file_extension = path.suffix
        full_path = str(path.absolute())

        return (parent_dir, file_stem, file_extension, full_path)



FILE_CLASS_MAPPINGS = {
    "KY_FilePathAnalyzer-": FilePathAnalyzer,
}

FILE_NAME_MAPPINGS = {
    "KY_FilePathAnalyzer-": "FilePath Analyzer",
}
