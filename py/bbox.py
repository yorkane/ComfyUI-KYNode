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

class BBoxPosition:
    """Determine which quadrant of an image a bounding box falls into"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "y": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "width": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "height": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "image_width": ("INT", {"default": 1920, "min": 1, "max": 10000}),
                "image_height": ("INT", {"default": 1080, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BBOX", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("position", "description", "bbox", "x", "y", "width", "height")
    FUNCTION = "get_position"
    CATEGORY = _CATEGORY

    def get_position(self, x, y, width, height, image_width, image_height):
        # Calculate the center point of the bounding box
        center_x = x + width / 2
        center_y = y + height / 2
        
        # Determine the quadrant based on the center point
        if center_x <= image_width / 2 and center_y <= image_height / 2:
            position = "top-left"
            description = "左上"
        elif center_x > image_width / 2 and center_y <= image_height / 2:
            position = "top-right"
            description = "右上"
        elif center_x <= image_width / 2 and center_y > image_height / 2:
            position = "bottom-left"
            description = "左下"
        else:  # center_x > image_width / 2 and center_y > image_height / 2
            position = "bottom-right"
            description = "右下"
            
        # Create a 512x512 bbox based on the quadrant
        bbox_size = 512
        if position == "top-left":
            bbox_x, bbox_y = 0, 0
        elif position == "top-right":
            bbox_x, bbox_y = image_width - bbox_size, 0
        elif position == "bottom-left":
            bbox_x, bbox_y = 0, image_height - bbox_size
        else:  # bottom-right
            bbox_x, bbox_y = image_width - bbox_size, image_height - bbox_size
            
        # Ensure bbox is within image bounds
        bbox_x = max(0, min(bbox_x, image_width - bbox_size))
        bbox_y = max(0, min(bbox_y, image_height - bbox_size))
        
        bbox = [bbox_x, bbox_y, bbox_size, bbox_size]
        
        return (position, description, bbox, bbox_x, bbox_y, bbox_size, bbox_size)


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




class XYWHtoBBox:
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
        }

    RETURN_TYPES = ("BBOX")
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
class BBoxToXYWH:
    """Convert x,y,width, height into bbox"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX", {"default": None})
            },
        }

    RETURN_TYPES = ("INT","INT","INT","INT")
    RETURN_NAMES = ("X", "Y", "Width", "Height",)
    FUNCTION = "convert"
    CATEGORY = _CATEGORY

    def convert(self, bbox):
        # if bbox is array of bboxes, get first
        if isinstance(bbox, (list, tuple)) and bbox and isinstance(bbox[0], (list, tuple)):
            if isinstance(bbox[0], (list, tuple)) and bbox and isinstance(bbox[0][0], (list, tuple)):
                bbox = bbox[0][0]
            else:
                bbox = bbox[0]
        return bbox[0], bbox[1], bbox[2], bbox[3]

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
    
class CreateMask:
    
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "createshapemask"
    CATEGORY = _CATEGORY
    DESCRIPTION = """
Creates a mask or batch of masks with the specified shape or bboxes.
bboxes input will override the shape and width/height parameters.
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
            "default": 'square'
             }),
                "frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
                "grow": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
        },
            "optional": {
                "bboxes1": ("BBOX",),
                "bboxes2": ("BBOX",),
                "location_x": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                "location_y": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                "shape_width": ("INT", {"default": 128,"min": 1, "max": 4096, "step": 1}),
                "shape_height": ("INT", {"default": 128,"min": 1, "max": 4096, "step": 1}),
            },
    } 

    def createshapemask(self, frames, frame_width, frame_height, location_x, location_y, shape_width, shape_height, grow, shape, bboxes1=None, bboxes2=None):
        # Define the number of images in the batch
        batch_size = frames
        out = []
        color = "white"
        
        # Handle bboxes input - could be single bbox or list of bboxes
        bbox_list1 = []
        bbox_list2 = []
        if bboxes1 is not None:
            # If it's a list of bboxes
            if isinstance(bboxes1, list) and len(bboxes1) > 0:
                # Check if it's a list of bboxes (list of lists)
                if isinstance(bboxes1[0], (list, tuple)):
                    bbox_list1 = bboxes1
                else:
                    # Single bbox passed as list
                    bbox_list1 = [bboxes1]
        if bboxes2 is not None:
            # If it's a list of bboxes
            if isinstance(bboxes2, list) and len(bboxes2) > 0:
                # Check if it's a list of bboxes (list of lists)
                if isinstance(bboxes2[0], (list, tuple)):
                    bbox_list2 = bboxes2
                else:
                    # Single bbox passed as list
                    bbox_list2 = [bboxes2]
        for i in range(batch_size):
            image = Image.new("RGB", (frame_width, frame_height), "black")
            draw = ImageDraw.Draw(image)

            # Calculate the size for this frame and ensure it's not less than 0


            # Draw shapes based on bboxes if provided
            if bbox_list1:
                for bbox in bbox_list1:
                    # Extract bbox coordinates
                    bbox_x, bbox_y, bbox_w, bbox_h = bbox
                    current_width = max(0, bbox_w + i*grow)
                    current_height = max(0, bbox_h + i*grow)
                    left_up_point = (bbox_x, bbox_y)
                    right_down_point = (bbox_x + bbox_w, bbox_y + bbox_h)
                    two_points = [left_up_point, right_down_point]
                    draw.rectangle(two_points, fill=color)
                for bbox in bbox_list2:
                    # Extract bbox coordinates
                    bbox_x, bbox_y, bbox_w, bbox_h = bbox
                    current_width = max(0, bbox_w + i*grow)
                    current_height = max(0, bbox_h + i*grow)
                    left_up_point = (bbox_x, bbox_y)
                    right_down_point = (bbox_x + bbox_w, bbox_y + bbox_h)
                    two_points = [left_up_point, right_down_point]
                    draw.rectangle(two_points, fill=color)
            else:
                current_width = max(0, shape_width + i*grow)
                current_height = max(0, shape_height + i*grow)
                # Original behavior when no bboxes provided
                if shape == 'circle' or shape == 'square':
                    # Define the bounding box for the shape
                    left_up_point = (location_x, location_y)
                    right_down_point = (location_x + current_width, location_y + current_height)
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

NODE_CLASS_MAPPINGS = {
    "KY_BBoxesToSAM2": BBoxesToSAM2,
    "KY_restoreBBox": restoreBBoxFrom,
    "KY_toBBox": XYWHtoBBox,
    "KY_BBoxToXYWH": BBoxToXYWH,
    "KY_JSONToBBox": JSONToBBox,
    "KY_ImageCropByBBox": ImageCropByBBox,
    "KY_CreateMask": CreateMask,
    "KY_BBoxPosition": BBoxPosition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KY_BBoxesToSAM2": "Prepare BBoxes for SAM2",
    "KY_restoreBBox": "restore bbox from VLM scale",
    "KY_toBBox": "Convert X Y WH into box",
    "KY_BBoxToXYWH": "Convert BBox to X Y Width Height",
    "KY_JSONToBBox": "json string to bbox",
    "KY_ImageCropByBBox": "Crop Image by bbox",
    "KY_CreateMask": "Create Mask by xywh",
    "KY_BBoxPosition": "BBox Image Position",
}
