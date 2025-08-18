import numpy as np
from PIL import Image

# 这个辅助函数与你原始代码中的完全相同，用于确定中间尺寸
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    Finds the closest aspect ratio from a list of target ratios.
    This function is identical to the one in the preprocessing script.
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def scale_bbox_to_original(bbox, orig_width, orig_height, image_size=448, min_num=1, max_num=12):
    """
    Scales a bounding box from the model's output coordinate system back to the original image's coordinate system.

    Args:
        bbox (list or tuple): The bounding box [xmin, ymin, xmax, ymax] from the model output.
        orig_size (tuple): The original image size as (width, height).
        image_size (int): The base processing size for each patch (e.g., 448).
        min_num (int): The min_num parameter used during preprocessing.
        max_num (int): The max_num parameter used during preprocessing.

    Returns:
        list: The absolute bounding box coordinates [xmin, ymin, xmax, ymax] on the original image.
    """

    model_xmin, model_ymin, model_xmax, model_ymax = bbox

    # --- 步骤 1: 重新计算预处理时的中间尺寸 (target_width, target_height) ---
    # 这部分逻辑必须与 `dynamic_preprocess` 函数完全一致，以确保尺寸计算同步。
    aspect_ratio = orig_width / orig_height
    
    # Generate the set of possible grid ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the best grid layout that matches the original aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate the dimensions of the intermediate resized image
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]

    # --- 步骤 2: 计算缩放比例 ---
    # 计算从“模型输出坐标系（中间大图）”到“原始图片坐标系”的缩放因子
    scale_w = orig_width / target_width
    scale_h = orig_height / target_height

    # --- 步骤 3: 应用缩放 ---
    # 将模型输出的 bbox 坐标乘以缩放因子，得到在原始图片上的绝对坐标
    final_xmin = int(round(model_xmin * aspect_ratio))
    final_ymin = int(round(model_ymin))
    final_xmax = int(round(model_xmax * aspect_ratio))
    final_ymax = int(round(model_ymax))

    return (final_xmin, final_ymin, final_xmax, final_ymax)