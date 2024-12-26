import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

import comfy.utils
import folder_paths

from .utils.image_convert import pil2tensor

_CATEGORY = 'KYNode/image'


class ReadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image_path': ('STRING', {'default': 'images'}),
            }
        }

    RETURN_TYPES = ('IMAGE', 'STRING')
    RETURN_NAMES = ('image', 'file_stem')
    FUNCTION = 'execute'
    CATEGORY = _CATEGORY
    DESCRIPTION = 'Read image from path'

    def execute(self, image_path):
        # 去掉可能存在的双引号
        image_path = image_path.strip('"')

        if not os.path.exists(image_path):
            raise FileNotFoundError(f'文件未找到: {image_path}')

        file_stem = str(Path(image_path).stem)

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        if img is None:
            raise ValueError(f'无法从文件中读取有效图像: {image_path}')

        if img.mode == 'I':
            img = img.point(lambda i: i * (1 / 255))
        img = img.convert('RGB')

        image = np.array(img).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image, file_stem)


class LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'input_path': ('STRING', {'default': '', 'multiline': False}),
                'start_index': ('INT', {'default': 0, 'min': 0, 'max': 9999}),
                'max_index': ('INT', {'default': 1, 'min': 1, 'max': 9999}),
                'step': ('INT', {'default': 1, 'min': 1, 'max': 100}),
            }
        }

    RETURN_TYPES = (
        'IMAGE',
        'STRING',
    )
    RETURN_NAMES = (
        'image_batch',
        'file_names',
    )
    OUTPUT_IS_LIST = (
        False,
        True,
    )
    FUNCTION = 'make_list'
    CATEGORY = _CATEGORY
    DESCRIPTION = 'read images from folder and return image batch and file names'

    def make_list(self, start_index, max_index, input_path, step):
        # 判断是否为绝对路径
        if os.path.isabs(input_path):
            full_input_dir = input_path
        else:
            # 相对路径则基于输出目录
            full_input_dir = os.path.join(folder_paths.get_output_directory(), input_path)

        # 检查输入路径是否存在
        if not os.path.exists(full_input_dir):
            raise FileNotFoundError(f'文件夹未找到: {full_input_dir}')

        # 检查文件夹是否为空
        if not os.listdir(full_input_dir):
            raise ValueError(f'文件夹为空: {full_input_dir}')

        # 对文件列表进行排序
        file_list = sorted(
            os.listdir(full_input_dir),
            key=lambda s: sum(((s, int(n)) for s, n in re.findall(r'(\D+)(\d+)', 'a%s0' % s)), ()),
        )

        image_list = []
        file_names = []  # 新增文件名列表

        # 确保 start_index 在列表范围内
        start_index = max(0, min(start_index, len(file_list) - 1))

        # 计算结束索引
        end_index = min(start_index + max_index, len(file_list))

        ref_image = None

        for num in range(start_index, end_index, step):
            fname = os.path.join(full_input_dir, file_list[num])
            file_names.append(file_list[num])  # 添加文件名到列表
            img = Image.open(fname)
            img = ImageOps.exif_transpose(img)
            if img is None:
                raise ValueError(f'无法从文件中读取有效图像: {fname}')
            image = img.convert('RGB')

            t_image = pil2tensor(image)
            # 确保所有图像的尺寸相同
            if ref_image is None:
                ref_image = t_image
            else:
                if t_image.shape[1:] != ref_image.shape[1:]:
                    t_image = comfy.utils.common_upscale(
                        t_image.movedim(-1, 1),
                        ref_image.shape[2],
                        ref_image.shape[1],
                        'lanczos',
                        'center',
                    ).movedim(1, -1)

            image_list.append(t_image)

        if not image_list:
            raise ValueError('未找到有效图像')

        image_batch = torch.cat(image_list, dim=0)

        return (
            image_batch,
            file_names,
        )


class KY_SaveImageToPath:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE",),
                     "img_template": ("STRING", {"default": "IMG-xx-######.png"}),
                     "start_index": ("INT", {"default": 1, "min": 0, "max": 999999}),
                     "quality": ("INT", {"default": 100, "min": 1, "max": 100}),
                     "extension": (["png", "webp", "jpg"],)},
                "optional": {
                    "lossless_webp": ("BOOLEAN", {"default": True}),
                    "optimize": ("BOOLEAN", {"default": False}),
                    "overwrite": ("BOOLEAN", {"default": True}),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save_image_to_path"
    OUTPUT_NODE = True
    CATEGORY = _CATEGORY
    DESCRIPTION = """save images to path, accept image batch and file template
    Default template: IMG-xx-######.png will save to output/IMG-xx-######.png
    ##### will be replaced by start_index auto-increment
    """

    def save_image_to_path(self, images, img_template="IMG-xx-######.png", 
                          start_index=1, quality=100, extension="png",
                          lossless_webp=True, optimize=False, overwrite=True,
                          prompt=None, extra_pnginfo=None):
        saved_paths = []
        
        # 分离目录和文件名模板
        template_dir = os.path.dirname(img_template)
        template_filename = os.path.basename(img_template)
        
        # 判断是否为绝对路径
        if os.path.isabs(template_dir):
            full_output_dir = template_dir
        else:
            # 相对路径则基于输出目录
            full_output_dir = os.path.join(folder_paths.get_output_directory(), template_dir)
        
        # 确保目录存在
        os.makedirs(full_output_dir, exist_ok=True)
        
        # 计算需要补零的位数
        zero_count = template_filename.count('#')
        name, ext = os.path.splitext(template_filename)
        
        # 保存所有图片
        for i, image in enumerate(images):
            # 生成文件名
            current_index = start_index + i
            number_str = str(current_index).zfill(zero_count)
            
            # 替换模板中的 # 为实际数字
            current_name = name.replace('#' * zero_count, number_str)
            current_filename = f"{current_name}.{extension}"
            current_path = os.path.join(full_output_dir, current_filename)

            # 检查文件是否存在且不允许覆盖
            if not overwrite and os.path.exists(current_path):
                raise ValueError(f'File exists could not overwrite: {current_path}')

            # 转换并保存图片
            img = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
            img.save(current_path,
                    quality=quality,
                    lossless=lossless_webp if extension == "webp" else None,
                    optimize=optimize)
            
            # 生成相对路径用于返回
            relative_path = os.path.join(template_dir, current_filename)
            saved_paths.append(relative_path)

        return {
            "ui": {"images": saved_paths},
            "result": (images,)
        }

IMG_CLASS_MAPPINGS = {
    "KY_ReadImage": ReadImage,
    "KY_LoadImagesFromFolder": LoadImagesFromFolder,
    "KY_SaveImageToPath": KY_SaveImageToPath
}

IMG_NAME_MAPPINGS = {
    "KY_ReadImage": "Read Image from Path",
    "KY_LoadImagesFromFolder": "Load Images From Folder",
    "KY_SaveImageToPath": "Save Image To Path"
}