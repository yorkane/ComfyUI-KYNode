import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
import concurrent.futures
import comfy.utils
import folder_paths

from .utils.image_convert import pil2tensor

_CATEGORY = 'KYNode/image'


def _natural_key(s):
    return sum(((s, int(n)) for s, n in re.findall(r'(\D+)(\d+)', 'a%s0' % s)), ())


class KY_Load_Images_from_path:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'input_dir': ('STRING', {'default': '', 'multiline': False}),
            },
            'optional': {
                'recursive': ('BOOLEAN', {'default': True}),
                'max_workers': ('INT', {'default': min(32, (os.cpu_count() or 1) + 4), 'min': 1, 'max': 128}),
                'limit': ('INT', {'default': 0, 'min': 0, 'max': 999999}),
                'shuffle': ('BOOLEAN', {'default': False}),
                'extensions': ('STRING', {'default': 'png,jpg,jpeg,webp,bmp,tif,tiff'}),
                'select_every_nth': ('INT', {'default': 1, 'min': 1, 'max': 999999}),
                'skip_first_images': ('INT', {'default': 0, 'min': 0, 'max': 999999}),
                'skip_last_images': ('INT', {'default': 0, 'min': 0, 'max': 999999}),
                'sort_mode': (['name_natural', 'name_lex', 'mtime_asc', 'mtime_desc'],),
                'keep_alpha_mask': ('BOOLEAN', {'default': True}),
            }
        }

    RETURN_TYPES = ('IMAGE', 'STRING', 'MASK')
    RETURN_NAMES = ('image_batch', 'file_names', 'mask_batch')
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = 'load_batch'
    CATEGORY = _CATEGORY
    DESCRIPTION = 'Multithreaded bulk image loader from a directory, returns a batched IMAGE and file names'

    @staticmethod
    def _is_valid_ext(file_path, allowed):
        ext = Path(file_path).suffix.lower().lstrip('.')
        return ext in allowed

    @staticmethod
    def _load_single(path, keep_alpha_mask=False):
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        if img.mode == 'I':
            img = img.point(lambda i: i * (1 / 255))
        mask_t = None
        if keep_alpha_mask and img.mode == 'RGBA':
            alpha = img.split()[-1]
            alpha_np = np.array(alpha).astype(np.float32) / 255.0
            mask_t = torch.from_numpy(alpha_np).unsqueeze(0)
        img_rgb = img.convert('RGB')
        return pil2tensor(img_rgb), mask_t

    def load_batch(self, input_dir, recursive=True, max_workers=8, limit=0, shuffle=False, extensions='png,jpg,jpeg,webp,bmp,tif,tiff', select_every_nth=1, skip_first_images=0, skip_last_images=0, sort_mode='name_natural', keep_alpha_mask=True):
        input_dir = input_dir.strip('"')

        if not input_dir:
            raise ValueError('input_dir is empty')

        if os.path.isabs(input_dir):
            full_dir = input_dir
        else:
            full_dir = os.path.join(folder_paths.get_input_directory(), input_dir)

        if not os.path.exists(full_dir):
            raise FileNotFoundError(f'Folder not found: {full_dir}')

        allowed_exts = {e.strip().lower() for e in extensions.split(',') if e.strip()}

        # Collect file paths
        file_paths = []
        if recursive:
            for root, _, files in os.walk(full_dir):
                for f in files:
                    p = os.path.join(root, f)
                    if self._is_valid_ext(p, allowed_exts):
                        file_paths.append(p)
        else:
            for f in os.listdir(full_dir):
                p = os.path.join(full_dir, f)
                if os.path.isfile(p) and self._is_valid_ext(p, allowed_exts):
                    file_paths.append(p)

        if not file_paths:
            raise ValueError(f'No images found in: {full_dir}')

        if sort_mode == 'name_lex':
            file_paths.sort(key=lambda p: os.path.basename(p))
        elif sort_mode == 'mtime_asc':
            file_paths.sort(key=lambda p: os.path.getmtime(p))
        elif sort_mode == 'mtime_desc':
            file_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        else:
            file_paths.sort(key=lambda p: _natural_key(os.path.basename(p)))
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(file_paths)

        total = len(file_paths)
        start = min(max(0, skip_first_images), total)
        end = max(0, total - max(0, skip_last_images))
        sel = file_paths[start:end]
        if select_every_nth > 1:
            sel = sel[::select_every_nth]
        if limit and limit > 0:
            sel = sel[:limit]
        file_paths = sel

        tensors = []
        masks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for img_mask in ex.map(lambda p: self._load_single(p, keep_alpha_mask), file_paths):
                img_t, mask_t = img_mask
                tensors.append(img_t)
                masks.append(mask_t)

        if not tensors:
            raise ValueError('Failed to load any images')

        ref = tensors[0]
        fixed = []
        fixed_masks = []
        for t in tensors:
            if t.shape[1:] != ref.shape[1:]:
                t_fixed = comfy.utils.common_upscale(
                    t.movedim(-1, 1),
                    ref.shape[2],
                    ref.shape[1],
                    'lanczos',
                    'center',
                ).movedim(1, -1)
                fixed.append(t_fixed)
            else:
                fixed.append(t)

        if any(m is not None for m in masks):
            for i, m in enumerate(masks):
                if m is None:
                    fixed_masks.append(None)
                    continue
                if m.shape[1:] != ref.shape[1:]:
                    m_in = m.unsqueeze(1)
                    m_fixed = comfy.utils.common_upscale(
                        m_in,
                        ref.shape[2],
                        ref.shape[1],
                        'lanczos',
                        'center',
                    ).squeeze(1)
                    fixed_masks.append(m_fixed)
                else:
                    fixed_masks.append(m)

        image_batch = torch.cat(fixed, dim=0)
        file_names = [os.path.relpath(p, folder_paths.get_input_directory()) if not os.path.isabs(input_dir) else os.path.basename(p) for p in file_paths]
        mask_batch = None
        if fixed_masks:
            present = [m for m in fixed_masks if m is not None]
            if present:
                h, w = image_batch.shape[1], image_batch.shape[2]
                zero = torch.zeros(1, h, w, dtype=image_batch.dtype)
                norm_masks = [m if m is not None else zero for m in fixed_masks]
                mask_batch = torch.cat(norm_masks, dim=0)

        return (image_batch, file_names, mask_batch)


NODE_CLASS_MAPPINGS = {
    'KY_Load_Images_from_path': KY_Load_Images_from_path,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'KY_Load_Images_from_path': 'KY Load Images from Path',
}
