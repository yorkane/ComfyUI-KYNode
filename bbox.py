import base64
import io
import json
import math
import os
import pprint
import random
import re
import time
import ast
import operator as op
import torch
from torch import Tensor
from torchvision import ops
from torchvision.transforms import functional
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from .utils.utility import pil2tensor, tensor2pil

_CATEGORY = "KYNode/BBox"


def is_integer(n):
    if n % 1 == 0:
        return True
    else:
        return False


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")

def safe_get_bbox(lst, index, default=None,  WrapArray=0):
    if 0 <= index < len(lst):
        if(WrapArray == 2):
            return [[lst[index]]]
        if(WrapArray == 1):
            return [lst[index]]
        else:
            return lst[index]
    return default

def is_deep_empty(obj):
    if obj is None:
        return True
    if isinstance(obj, str):
        return obj.strip() == ''
    if isinstance(obj, (list, tuple, set)):
        if len(obj) == 0:
            return True
        return all(is_deep_empty(item) for item in obj)
    if isinstance(obj, dict):
        return len(obj) == 0 or all(is_deep_empty(v) for v in obj.values())
    if hasattr(obj, 'numel'):  # PyTorch Tensor
        return obj.numel() == 0
    if hasattr(obj, 'size'):   # NumPy Array
        return obj.size == 0
    return False  # 非空值（数字、非空字符串、对象等）




class JSONToBBox:
    """Convert a list of bounding boxes to the format expected by SAM2 nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        
        return {
            "optional": {
                "jsonBbox": ("JSON", {"default": ""}),
                "bboxes": ("BBOX", {"default": []}),
                "bboxJSONKey": ("STRING", {"default": "bbox_2d"}),
            }
        }

    RETURN_TYPES = ("BBOX_LIST","BBOX","BBOX","BBOX", )
    RETURN_NAMES = ("bbox list","bbox1","bbox2","bbox3", )
    FUNCTION = "convert"
    CATEGORY = _CATEGORY

    def convert(self, jsonBbox, bboxes=[[0,0,0,0]], bboxJSONKey=""):
        if jsonBbox == "":
            box_2d = bboxes
        else:
            box_2d = json.loads(jsonBbox)
        if not isinstance(box_2d, list):
            raise ValueError("bboxes must be a list")
        bboxes = [b[bboxJSONKey] for b in box_2d]

        return (bboxes, safe_get_bbox(bboxes,0, None), safe_get_bbox(bboxes,1, None,), safe_get_bbox(bboxes,2, None) )




class BBoxesToSAM2:
    """Convert a list of bounding boxes to the format expected by SAM2 nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        
        return {
            "optional": {
                "jsonBbox": ("JSON", {"default": ""}),
                "bboxes": ("BBOX", {"default": [[[]]]})
            }
        }

    RETURN_TYPES = ("BBOXES","BBOXES","BBOXES","BBOXES", )
    RETURN_NAMES = ("All sam2_bboxes","sam2_bboxe1","sam2_bboxe2","sam2_bboxe3", )
    FUNCTION = "convert"
    CATEGORY = _CATEGORY

    def convert(self, jsonBbox, bboxes=[[[0,0,0,0]]]):
        if jsonBbox == "":
            box_2d = bboxes
        else:
            box_2d = json.loads(jsonBbox)
        if not isinstance(box_2d, list):
            raise ValueError("bboxes must be a list")
        bboxes = [b["bbox_2d"] for b in box_2d]

        # If already batched, return as-is
        if bboxes and isinstance(bboxes[0], (list, tuple)) and bboxes[0] and isinstance(bboxes[0][0], (list, tuple)):
            return (bboxes, [safe_get_bbox(bboxes,0, [],1)], [safe_get_bbox(bboxes,1, [])], [safe_get_bbox(bboxes,2, [])])

        return ([bboxes], safe_get_bbox(bboxes,0, None, 2), safe_get_bbox(bboxes,1, None, 2), safe_get_bbox(bboxes,2, None, 2))




class toBBox:
    """Convert x,y,width, height into bbox"""
    @classmethod
    def INPUT_TYPES(cls):
        
        return {
            "required": {
                "X": ("INT", {"default": 0}),
                "Y": ("INT", {"default": 0}),
                "Width": ("INT", {"default": 1920}),
                "Height": ("INT", {"default": 1080}),
                "WrapArray": ("INT", {"default": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("BBOX", "BBOXES", "STRING")
    RETURN_NAMES = ("bbox",)
    FUNCTION = "convert"
    CATEGORY = _CATEGORY

    def convert(self, X, Y, Width, Height, WrapArray=1):
        bbox = [X, Y, Width, Height]
        if(WrapArray == 3):
            return ([[bbox]],)
        if(WrapArray == 2):
            return ([bbox],)
        return (bbox,)

from .utils.vlm_bbox import scale_bbox_to_original
class restoreBBoxFrom:
    """Restore bbox to original image scale"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX", {"default": [0,0,0,0]}),
                "originWidth": ("INT", {"default": 1920}),
                "originHeight": ("INT", {"default": 1080}),
                "gridSize": ("INT", {"default": 448}),
                "minGridNum": ("INT", {"default": 1}),
                "maxGridNum": ("INT", {"default": 12}),
            },
        }
    RETURN_TYPES = ("BBOX","INT","INT","INT", "INT")
    RETURN_NAMES = ("bbox", "x", "y" , "width", "height")
    FUNCTION = "convert"
    CATEGORY = _CATEGORY

    def convert(self, bbox, originWidth, originHeight, gridSize, minGridNum, maxGridNum):
        x,y,w,h = scale_bbox_to_original(
            bbox=bbox,
            orig_width=originWidth,
            orig_height=originHeight,
            image_size=gridSize,
            max_num=maxGridNum,
            min_num=1,
        )
        abs_bbox = [x,y,w,h]
        abs_bbox = [int(round(coord)) for coord in abs_bbox]
        return (abs_bbox, x,y,w,h)


class ImageCropByBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX", {}),
                "image": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "main"

    CATEGORY = _CATEGORY

    def main(self, bbox: Tensor, image: Tensor):
        results = []
        image_permuted = image.permute(0, 3, 1, 2)
        for image_item in image_permuted:
            x = bbox[0]
            y = bbox[1]
            width = bbox[2]
            height = bbox[3]
            cropped_image = functional.crop(image_item, y, x, height, width) # type: ignore
            result = cropped_image.permute(1, 2, 0)
            results.append(result)
        try: 
            results = torch.stack(results, dim=0)
        except:
            pass
        return (results,)
    
class CreateShapeMask:
    
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "createshapemask"
    CATEGORY = "KJNodes/masking/generate"
    DESCRIPTION = """
Creates a mask or batch of masks with the specified shape.  
Locations are center locations.  
Grow value is the amount to grow the shape on each frame, creating animated masks.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shape": (
            [   'circle',
                'square',
                'triangle',
            ],
            {
            "default": 'circle'
             }),
                "frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
                "location_x": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                "location_y": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                "grow": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "shape_width": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
                "shape_height": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
        },
    } 

    def createshapemask(self, frames, frame_width, frame_height, location_x, location_y, shape_width, shape_height, grow, shape):
        # Define the number of images in the batch
        batch_size = frames
        out = []
        color = "white"
        for i in range(batch_size):
            image = Image.new("RGB", (frame_width, frame_height), "black")
            draw = ImageDraw.Draw(image)

            # Calculate the size for this frame and ensure it's not less than 0
            current_width = max(0, shape_width + i*grow)
            current_height = max(0, shape_height + i*grow)

            if shape == 'circle' or shape == 'square':
                # Define the bounding box for the shape
                left_up_point = (location_x, location_y) # bottom left
                right_down_point = (location_x + current_width, location_y + current_height) # bottom right
                two_points = [left_up_point, right_down_point]

                if shape == 'circle':
                    draw.ellipse(two_points, fill=color)
                elif shape == 'square':
                    draw.rectangle(two_points, fill=color)
                    
            elif shape == 'triangle':
                # Define the points for the triangle
                left_up_point = (location_x - current_width // 2, location_y + current_height // 2) # bottom left
                right_down_point = (location_x + current_width // 2, location_y + current_height // 2) # bottom right
                top_point = (location_x, location_y - current_height // 2) # top point
                draw.polygon([top_point, left_up_point, right_down_point], fill=color)

            image = pil2tensor(image)
            mask = image[:, :, :, 0]
            out.append(mask)
        outstack = torch.cat(out, dim=0)
        return (outstack, 1.0 - outstack,)

BBOX_NODE_CLASS_MAPPINGS = {
    "KY_BBoxesToSAM2": BBoxesToSAM2,
    "KY_restoreBBox": restoreBBoxFrom,
    "KY_toBBox": toBBox,
    "KY_JSONToBBox": JSONToBBox,
    "KY_ImageCropByBBox": ImageCropByBBox,
    "KY_CreateMask": CreateShapeMask,
    
}

BBOX_NODE_NAME_MAPPINGS = {
    "KY_BBoxesToSAM2": "Prepare BBoxes for SAM2",
    "KY_restoreBBox": "restore bbox from VLM scale",
    "KY_toBBox": "Convert X Y WH into box",
    "KY_JSONToBBox": "json string to bbox",
    "KY_ImageCropByBBox": "Crop Image by bbox",
    "KY_CreateMask": "Crop Mask by xywh",
}
