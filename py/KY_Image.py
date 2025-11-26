import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
import concurrent.futures
import comfy.utils
import folder_paths
from .utils.net_io import load_images_from_url
from .utils.image_convert import pil2tensor, tensor2pil

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

class KY_SaveImageToPath:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", {"tooltip": "Input image batch to be saved."}),
                     "img_template": ("STRING", {"default": "IMG-######", "tooltip": "Filename template. Use '#' for sequence numbering (e.g., 'IMG-######'). Supports relative folders (e.g., 'my_folder/img_###')."}),
                     "start_index": ("INT", {"default": 1, "min": 0, "max": 999999, "tooltip": "Starting number for the sequence (replaces '######' in the template)."}),
                     "extension": (["png", "webp", "jpg"], {"tooltip": "Output file format."}),
                     # 合并后的通用质量参数 (适用于 JPEG 和 WebP)
                     "quality": ("INT", {"default": 95, "min": 1, "max": 100, "tooltip": "Quality for JPEG and WebP (1-100). Higher is better quality. Ignored for PNG."}), 
                     # PNG 专用压缩等级 (0=快/大, 9=慢/小)
                     "png_compression": ("INT", {"default": 1, "min": 0, "max": 9, "tooltip": "Compression level for PNG (0-9). 0 is fastest (large file), 9 is slowest (small file). 1 is recommended for speed."}), 
                     },
                "optional": {
                    # WebP 专用无损开关
                    "webp_lossless": ("BOOLEAN", {"default": False, "tooltip": "Enable WebP lossless mode. If True, quality is forced to 100. Encode speed will be slower."}),
                    "overwrite": ("BOOLEAN", {"default": True, "tooltip": "If True, overwrites existing files. If False, adds a counter suffix (e.g., -[0001]) to the filename."}),
                    "NOTSAVE": ("BOOLEAN", {"default": False, "tooltip": "If True, skips the saving process but passes the images through. Useful for debugging logic without writing disk."}),
                    "show_preview": ("BOOLEAN", {"default": True, "tooltip": "If True, displays the saved images in the node UI. Disable this for maximum performance with large batches."}),
                },
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save_image_cv2"
    OUTPUT_NODE = True
    CATEGORY = "KY_Nodes"
    DESCRIPTION = """Ultra-fast image saver using OpenCV (cv2). 
    
    Features:
    - Multithreaded saving
    - Supports PNG, JPEG, WebP
    - Customizable compression/quality
    - No metadata saving (for speed)
    """

    @staticmethod
    def save_single_file_cv2(image_np, file_path, extension, quality, png_compression, webp_lossless):
        try:
            # 1. 格式转换 (ComfyUI RGB -> OpenCV BGR)
            # image_np 是 float32 (0-1)，需要先乘 255 转 uint8
            img_uint8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            
            # 2. 准备编码参数
            encode_params = []
            if extension == "png":
                # CV2_IMWRITE_PNG_COMPRESSION: 0-9 (默认为1，追求速度)
                encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression]
            
            elif extension == "webp":
                # CV2_IMWRITE_WEBP_QUALITY: 1-100
                # 如果开启无损，OpenCV 通常需要高质量设置
                current_quality = 100 if webp_lossless else quality
                encode_params = [int(cv2.IMWRITE_WEBP_QUALITY), current_quality]
            
            elif extension == "jpg":
                # CV2_IMWRITE_JPEG_QUALITY: 0-100
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

            # 3. 编码并写入
            # 使用 imencode 解决中文/特殊字符路径问题
            success, buffer = cv2.imencode(f".{extension}", img_bgr, encode_params)
            
            if success:
                with open(file_path, "wb") as f:
                    f.write(buffer)
                return True
            return False
            
        except Exception as e:
            print(f"Error saving {file_path}: {e}")
            return False

    def save_image_cv2(self, images, img_template="IMG-####", 
                       start_index=1, extension="png",
                       quality=95, png_compression=1, webp_lossless=False,
                       overwrite=True, NOTSAVE=False, show_preview=True):
        if NOTSAVE:
            return {"ui": {"images": []}, "result": (images,)}

        # 路径处理
        template_dir = os.path.dirname(img_template)
        template_filename = os.path.basename(img_template)
        
        if os.path.isabs(template_dir):
            full_output_dir = template_dir
        else:
            full_output_dir = os.path.join(folder_paths.get_output_directory(), template_dir)
        
        os.makedirs(full_output_dir, exist_ok=True)
        
        zero_count = template_filename.count('#')
        name_base, _ = os.path.splitext(template_filename)
        
        # 数据准备 (Tensor -> Numpy)
        images_np = images.cpu().numpy()
        
        tasks = []
        saved_paths = []
        
        # 预计算文件名
        for i in range(len(images_np)):
            if zero_count > 0:
                current_index = start_index + i
                number_str = str(current_index).zfill(zero_count)
                current_name = name_base.replace('#' * zero_count, number_str)
            else:
                current_name = name_base
            
            current_filename = f"{current_name}.{extension}"
            current_path = os.path.join(full_output_dir, current_filename)
            
            # 覆盖/重命名逻辑
            if not overwrite and os.path.exists(current_path):
                counter = 1
                base_n, file_e = os.path.splitext(current_filename)
                while os.path.exists(current_path):
                    if counter > 9999:
                         raise ValueError(f'Suffix limit exceeded: {current_path}')
                    current_filename = f"{base_n}-[{str(counter).zfill(4)}]{file_e}"
                    current_path = os.path.join(full_output_dir, current_filename)
                    counter += 1
            
            # 任务参数: (数据, 路径, 扩展名, 质量, PNG压缩等级, WebP无损开关)
            tasks.append((images_np[i], current_path, extension, quality, png_compression, webp_lossless))
            saved_paths.append(os.path.join(template_dir, current_filename))

        # 多线程执行
        # CV2 释放 GIL 做得很好，因此可以跑满 CPU
        max_workers = min(32, (os.cpu_count() or 1) + 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.save_single_file_cv2, *task) 
                for task in tasks
            ]
            concurrent.futures.wait(futures)

        # 构建返回结果
        result_dict = {"result": (images,)}
        
        if show_preview:
            result_dict["ui"] = {"images": [{"filename": p, "type": "output", "subfolder": ""} for p in saved_paths]}
            
        return result_dict
            
        return result_dict
class KY_LoadImageFrom:
    _CATEGORY = 'KYNode/image'  # Using the common category from the file

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False, "placeholder": "Enter path, URL, or Base64 data (supports multiple URLs separated by newlines)"}),
            },
            "optional": {
                 "input_image": ("IMAGE",),
                 "input_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "execute_load"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True

    def _process_pil_image_to_tensors(self, pil_image):
        if pil_image is None:
            return None, None
        
        mask_tensor = None
        if pil_image.mode == 'RGBA':
            alpha = pil_image.split()[-1]
            mask_pil = Image.new('L', pil_image.size)
            mask_pil.putalpha(alpha) 
            # pil2tensor for 'L' mode gives (1, H, W, 1). We want (1, H, W) for the mask.
            mask_tensor = pil2tensor(mask_pil)[0, ..., 0].unsqueeze(0) # Results in (1, H, W)
        
        img_pil_rgb = pil_image.convert('RGB')
        image_tensor = pil2tensor(img_pil_rgb) # Results in (1, H, W, C) for RGB
        return image_tensor, mask_tensor

    def execute_load(self, input_string="", input_image=None, input_mask=None):
        loaded_images = []
        loaded_masks = []

        if input_string and input_string.strip():
            # 支持多行输入，每行一个URL/路径/Base64
            lines = [line.strip() for line in input_string.strip().split('\n') if line.strip()]
            
            try:
                # 使用net_io中的load_images_from_url方法统一处理各种输入类型
                pil_images, pil_masks = load_images_from_url(lines, keep_alpha_channel=True)
                
                for i, pil_image in enumerate(pil_images):
                    if pil_image is not None:
                        image_tensor, mask_tensor = self._process_pil_image_to_tensors(pil_image)
                        if image_tensor is not None:
                            loaded_images.append(image_tensor)
                            loaded_masks.append(mask_tensor)
            except Exception as e:
                print(f"Warning: Failed to load from input_string: {e}")

        # Fallback to input_image if string loading failed or no string provided
        if not loaded_images and input_image is not None:
            loaded_images.append(input_image)
            loaded_masks.append(None)

        if not loaded_images:
            raise ValueError("KY_LoadImageFrom: No valid image. Provide path, URL, Base64, or connect image.")
        
        # 处理mask
        final_masks = []
        for i, loaded_mask in enumerate(loaded_masks):
            final_mask = None
            if loaded_mask is not None:
                final_mask = loaded_mask
            elif input_mask is not None:
                final_mask = input_mask
                # 如果输入mask只有一个但图片有多个，需要复制mask
                if i > 0 and input_mask.shape[0] == 1:
                    if (input_mask.shape[1] == loaded_images[i].shape[1] and
                        input_mask.shape[2] == loaded_images[i].shape[2]):
                        final_mask = input_mask
                    else:
                        print(f"Warning: KY_LoadImageFrom - input_mask dimensions mismatch with image {i} dimensions.")
            final_masks.append(final_mask)

        # 处理多个图片：确保每个图片都有正确的维度
        processed_images = []
        processed_masks = []
        
        for img_tensor in loaded_images:
            # 确保张量维度正确：应该是 (1, H, W, C)
            while img_tensor.dim() > 4:
                img_tensor = img_tensor.squeeze(0)
            if img_tensor.dim() == 3:  # (H, W, C)
                img_tensor = img_tensor.unsqueeze(0)  # -> (1, H, W, C)
            elif img_tensor.dim() == 4 and img_tensor.shape[0] != 1:
                # 如果第一个维度不是1，可能需要重新整理
                img_tensor = img_tensor.squeeze(0).unsqueeze(0)
            processed_images.append(img_tensor)
        
        for mask_tensor in final_masks:
            if mask_tensor is not None:
                # 确保mask维度正确：应该是 (1, H, W)
                while mask_tensor.dim() > 3:
                    mask_tensor = mask_tensor.squeeze(0)
                if mask_tensor.dim() == 2:  # (H, W)
                    mask_tensor = mask_tensor.unsqueeze(0)  # -> (1, H, W)
                elif mask_tensor.dim() == 3 and mask_tensor.shape[0] != 1:
                    mask_tensor = mask_tensor.squeeze(0).unsqueeze(0)
                processed_masks.append(mask_tensor)
            else:
                processed_masks.append(None)
        # 始终返回列表格式，匹配OUTPUT_IS_LIST配置
        return (processed_images, processed_masks)


class CropImageByXYWH:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "y": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "width": ("INT", {"default": 512, "min": 64, "max": 999999}),
                "height": ("INT", {"default": 512, "min": 64, "max": 999999}),
                "divisible_by": ("INT", {"default": 1, "min": 1, "max": 128}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Crop image by XYWH coordinates with optional divisible size adjustment"

    def crop(self, image, x, y, width, height, divisible_by):
        # 确保最小尺寸为64
        width = max(64, width)
        height = max(64, height)
        
        # 优先确保尺寸能被divisible_by整除（只能放大，不能缩小）
        if divisible_by > 1:
            # 使用向上取整的方式计算新的尺寸
            adjusted_width = ((width + divisible_by - 1) // divisible_by) * divisible_by
            adjusted_height = ((height + divisible_by - 1) // divisible_by) * divisible_by
        else:
            adjusted_width = width
            adjusted_height = height
            
        # 获取图像尺寸
        batch_size, img_height, img_width, channels = image.shape
        
        # 保持XY起始点不变
        x_start = x
        y_start = y
        
        # 计算结束点
        x_end = x_start + adjusted_width
        y_end = y_start + adjusted_height
        
        # 如果裁剪区域超出右边界，则调整宽度
        if x_end > img_width:
            adjusted_width = img_width - x_start
            # 确保调整后的宽度仍然满足divisible_by要求
            if divisible_by > 1:
                adjusted_width = (adjusted_width // divisible_by) * divisible_by
            x_end = x_start + adjusted_width
            
        # 如果裁剪区域超出下边界，则调整高度
        if y_end > img_height:
            adjusted_height = img_height - y_start
            # 确保调整后的高度仍然满足divisible_by要求
            if divisible_by > 1:
                adjusted_height = (adjusted_height // divisible_by) * divisible_by
            y_end = y_start + adjusted_height
            
        # 确保裁剪区域有效
        x_end = min(img_width, x_end)
        y_end = min(img_height, y_end)
        
        # 确保裁剪区域尺寸大于0
        if x_end <= x_start or y_end <= y_start:
            raise ValueError("Crop area is invalid. Check your parameters.")
        
        # 执行裁剪
        cropped_image = image[:, y_start:y_end, x_start:x_end, :]
        
        return (cropped_image,)


NODE_CLASS_MAPPINGS = {
    "KY_ReadImage": ReadImage,
    "KY_SaveImageToPath": KY_SaveImageToPath,
    "KY_LoadImageFrom": KY_LoadImageFrom,
    "KY_CropImageByXYWH": CropImageByXYWH
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KY_ReadImage": "Read Image from Path",
    "KY_SaveImageToPath": "Save Images To Path with sequence number",
    "KY_LoadImageFrom": "Load Image (Path/URL/Base64/Input)",
    "KY_CropImageByXYWH": "Crop Image by XYWH"
}