import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

import comfy.utils

from .utils.image_convert import pil2tensor

_CATEGORY = "KYNode/files"


class ReadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "images"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "file_stem")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "读取指定路径图片，返回图片和图片名称"

    def execute(self, image_path):
        # 去掉可能存在的双引号
        image_path = image_path.strip('"')

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"文件未找到: {image_path}")

        file_stem = str(Path(image_path).stem)

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        if img is None:
            raise ValueError(f"无法从文件中读取有效图像: {image_path}")

        if img.mode == "I":
            img = img.point(lambda i: i * (1 / 255))
        img = img.convert("RGB")

        image = np.array(img).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image, file_stem)


class LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": "", "multiline": False}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "max_index": ("INT", {"default": 1, "min": 1, "max": 9999}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "images_list",
        "image_batch",
    )
    OUTPUT_IS_LIST = (
        True,
        False,
    )
    FUNCTION = "make_list"
    CATEGORY = _CATEGORY
    DESCRIPTION = "读取文件夹中的图片，返回图片列表和图片批次"

    def make_list(self, start_index, max_index, input_path):
        # 检查输入路径是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件夹未找到: {input_path}")

        # 检查文件夹是否为空
        if not os.listdir(input_path):
            raise ValueError(f"文件夹为空: {input_path}")

        # 对文件列表进行排序
        file_list = sorted(
            os.listdir(input_path),
            key=lambda s: sum(
                ((s, int(n)) for s, n in re.findall(r"(\D+)(\d+)", "a%s0" % s)), ()
            ),
        )

        image_list = []

        # 确保 start_index 在列表范围内
        start_index = max(0, min(start_index, len(file_list) - 1))

        # 计算结束索引
        end_index = min(start_index + max_index, len(file_list))

        ref_image = None

        for num in range(start_index, end_index):
            fname = os.path.join(input_path, file_list[num])
            img = Image.open(fname)
            img = ImageOps.exif_transpose(img)
            if img is None:
                raise ValueError(f"无法从文件中读取有效图像: {fname}")
            image = img.convert("RGB")

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
                        "lanczos",
                        "center",
                    ).movedim(1, -1)

            image_list.append(t_image)

        if not image_list:
            raise ValueError("未找到有效图像")

        image_batch = torch.cat(image_list, dim=0)
        images_out = [image_batch[i : i + 1, ...] for i in range(image_batch.shape[0])]

        return (
            images_out,
            image_batch,
        )


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


class KY_SaveImageToPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "IMG": ("IMAGE",),
                "full_file_path": ("STRING", {"default": "./ComfyUI.png"}),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100}),
                "extension": (["png", "webp", "jpg"],),
            },
            "optional": {
                "lossless_webp": ("BOOLEAN", {"default": True}),
                "optimize": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save_image_to_path"
    OUTPUT_NODE = True
    CATEGORY = _CATEGORY

    def save_image_to_path(
        self,
        IMG,
        full_file_path="ComfyUI.png",
        quality=100,
        lossless_webp=True,
        optimize=False,
        extension="webp",
        prompt=None,
        extra_pnginfo=None,
    ):
        # results = self.save_images(images, filename_prefix, prompt, extra_pnginfo)
        saved_paths = []
        # folder_structure = []
        # folder_structure = json.loads(folder_structure)
        base_directory = os.path.dirname(full_file_path)
        file_name_without_extension = os.path.splitext(
            os.path.basename(full_file_path)
        )[0]
        target_file_path = (
            base_directory + "/" + file_name_without_extension + "." + extension
        )
        images = IMG
        # Ensure base directory exists
        # os.makedirs(base_directory, exist_ok=True)

        for i, image in enumerate(images):
            # Convert the image tensor to a PIL Image
            img = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))

            # Create the full folder path based on the folder structure
            # full_folder_path = self.create_folder_path(base_directory, [])
            os.makedirs(base_directory, exist_ok=True)

            # Create the file name and ensure it doesn't overwrite existing files
            # index = 0 + i
            # while True:
            # full_file_name = file_name_template.format(index=index)
            #    full_file_path = path #os.path.join(full_folder_path, full_file_name)
            #    if not os.path.exists(full_file_path):
            #        break
            #    index += 1

            # Save the image
            img.save(
                target_file_path,
                quality=quality,
                lossless=lossless_webp,
                optimize=optimize,
            )

            # Save metadata if provided
            # if metadata:
            #    metadata_file_name = f"{os.path.splitext(full_file_name)[0]}_metadata.txt"
            #    metadata_file_path = os.path.join(full_folder_path, metadata_file_name)
            #    with open(metadata_file_path, 'w') as f:
            #        f.write(metadata)

            saved_paths.append(target_file_path)
            break
        return {"ui": {"images": target_file_path}, "result": (IMG,)}


FILE_CLASS_MAPPINGS = {
    "KY_ReadImage-": ReadImage,
    "KY_LoadImagesFromFolder-": LoadImagesFromFolder,
    "KY_FilePathAnalyzer-": FilePathAnalyzer,
    "KY_SaveImageToPath": KY_SaveImageToPath,
}

FILE_NAME_MAPPINGS = {
    "KY_ReadImage-": "Read Image from Path",
    "KY_LoadImagesFromFolder-": "Load Images From Folder",
    "KY_FilePathAnalyzer-": "FilePath Analyzer",
    "KY_SaveImageToPath": "Save Single Image in full path",
}
