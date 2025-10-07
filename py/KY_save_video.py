# KY_save_video 保存视频的 ComfyUI节点
# 用于将包含alpha 通道的PNG图片转换保存为 mov 格式的视频，保证输出视频包含alpha 通道

import os
import subprocess
import tempfile
import torch
import numpy as np
from PIL import Image, ImageOps
import time
import uuid
from pathlib import Path

import folder_paths
from .utils.logger import logger
from .utils.video_utils import ffmpeg_path, ENCODE_ARGS, BIGMAX, DIMMAX


class KY_SaveVideo:
    """保存图像序列为支持alpha通道的视频文件"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "KY_video"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.1}),
                "quality": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),
                "codec": (["prores_ks", "png", "qtrle"], {"default": "prores_ks"}),
                "format": (["mov", "avi", "mp4"], {"default": "mov"}),
            },
            "optional": {
                "mask": ("MASK",),
                "audio": ("AUDIO",),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": BIGMAX}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "preview_url")
    FUNCTION = "save_video"
    CATEGORY = "KY_Video"
    OUTPUT_NODE = True
    
    def save_video(self, images, filename_prefix, fps, quality, codec, format, 
                   mask=None, audio=None, start_frame=0, end_frame=-1, loop_count=0):
        """
        保存图像序列为视频文件
        
        Args:
            images: 输入图像张量 (B, H, W, C)
            filename_prefix: 文件名前缀
            fps: 帧率
            quality: 视频质量 (CRF值，越小质量越高)
            codec: 视频编码器
            format: 输出格式
            mask: 可选的alpha通道蒙版
            audio: 可选的音频数据
            start_frame: 起始帧
            end_frame: 结束帧 (-1表示到最后)
            loop_count: 循环次数
        """
        
        if ffmpeg_path is None:
            raise Exception("FFmpeg not found. Please install FFmpeg to use video save functionality.")
        
        # 处理帧范围
        total_frames = len(images)
        if end_frame == -1:
            end_frame = total_frames
        else:
            end_frame = min(end_frame, total_frames)
        
        if start_frame >= end_frame:
            raise ValueError(f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}")
        
        # 选择帧范围
        selected_images = images[start_frame:end_frame]
        selected_mask = None
        if mask is not None:
            selected_mask = mask[start_frame:end_frame]
        
        logger.info(f"Saving {len(selected_images)} frames as {format} video with {codec} codec")
        
        # 创建输出目录
        output_dir = folder_paths.get_output_directory()
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"{filename_prefix}_{timestamp}_{unique_id}.{format}"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # 根据编码器选择不同的保存方式
            if codec == "png":
                self._save_with_png_codec(selected_images, selected_mask, 
                                        output_path, fps, audio, loop_count)
            else:
                self._save_with_video_codec(selected_images, selected_mask,
                                           output_path, fps, quality, codec, format, audio, loop_count)
            
            # 生成预览URL
            preview_url = f"/view?filename={output_filename}&type=output&format=video/{format}"
            
            logger.info(f"Video saved successfully: {output_path}")
            return (output_path, preview_url)
            
        except Exception as e:
            logger.error(f"Failed to save video: {str(e)}")
            raise e
    
    def _save_with_png_codec(self, images, mask, output_path, fps, audio, loop_count):
        """使用PNG编码器保存，适合保留alpha通道"""
        
        # 创建临时目录存储PNG序列
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存每一帧为PNG
            for i, img_tensor in enumerate(images):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                
                # 转换张量到PIL图像
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np, mode='RGB')
                
                # 如果有mask，添加alpha通道
                if mask is not None:
                    alpha_np = (mask[i].cpu().numpy() * 255).astype(np.uint8)
                    img_pil = img_pil.convert('RGBA')
                    img_pil.putalpha(Image.fromarray(alpha_np, mode='L'))
                
                img_pil.save(frame_path, "PNG")
            
            # 使用ffmpeg将PNG序列转换为视频
            input_pattern = os.path.join(temp_dir, "frame_%06d.png")
            self._ffmpeg_encode(input_pattern, output_path, fps, "png", None, "mov", audio, loop_count, is_sequence=True)
    
    def _save_with_video_codec(self, images, mask, output_path, fps, quality, codec, format, audio, loop_count):
        """使用视频编码器保存"""
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 如果有alpha通道，合并到图像中
            if mask is not None:
                # 将RGB图像和alpha mask合并为RGBA
                rgba_images = []
                for i in range(len(images)):
                    rgb = images[i].cpu().numpy()  # (H, W, 3)
                    alpha = mask[i].cpu().numpy()  # (H, W)
                    rgba = np.concatenate([rgb, alpha[..., np.newaxis]], axis=-1)  # (H, W, 4)
                    rgba_images.append(rgba)
                
                # 保存为带alpha的临时视频
                temp_raw_path = os.path.join(temp_dir, "temp_rgba.raw")
                self._save_raw_rgba(rgba_images, temp_raw_path)
                
                # 使用ffmpeg编码
                height, width = rgba_images[0].shape[:2]
                self._ffmpeg_encode_raw_rgba(temp_raw_path, output_path, width, height, fps, quality, codec, format, audio, loop_count)
            else:
                # 只有RGB，保存为临时PNG序列然后编码
                for i, img_tensor in enumerate(images):
                    frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                    img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np, mode='RGB')
                    img_pil.save(frame_path, "PNG")
                
                input_pattern = os.path.join(temp_dir, "frame_%06d.png")
                self._ffmpeg_encode(input_pattern, output_path, fps, codec, quality, format, audio, loop_count, is_sequence=True)
    
    def _save_raw_rgba(self, rgba_images, output_path):
        """保存RGBA图像序列为原始数据"""
        with open(output_path, 'wb') as f:
            for rgba in rgba_images:
                # 转换为uint8并写入
                rgba_uint8 = (rgba * 255).astype(np.uint8)
                f.write(rgba_uint8.tobytes())
    
    def _ffmpeg_encode_raw_rgba(self, input_path, output_path, width, height, fps, quality, codec, format, audio, loop_count):
        """从原始RGBA数据编码视频"""
        
        cmd = [ffmpeg_path]
        
        # 输入参数
        cmd.extend([
            '-f', 'rawvideo',
            '-pixel_format', 'rgba',
            '-video_size', f'{width}x{height}',
            '-framerate', str(fps),
            '-i', input_path
        ])
        
        # 添加音频输入
        if audio is not None:
            audio_path = self._save_temp_audio(audio)
            cmd.extend(['-i', audio_path])
        
        # 编码参数
        if codec == "prores_ks":
            cmd.extend(['-c:v', 'prores_ks', '-profile:v', '4444', '-pix_fmt', 'yuva444p10le'])
        elif codec == "qtrle":
            cmd.extend(['-c:v', 'qtrle', '-pix_fmt', 'rgba'])
        
        if quality is not None and codec not in ["png", "qtrle"]:
            cmd.extend(['-crf', str(quality)])
        
        # 循环参数
        if loop_count > 0:
            cmd.extend(['-loop', str(loop_count)])
        
        # 音频参数
        if audio is not None:
            cmd.extend(['-c:a', 'aac'])
        
        # 输出
        cmd.extend(['-y', output_path])
        
        logger.info(f"FFmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            logger.info("FFmpeg encoding completed successfully")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode(*ENCODE_ARGS) if e.stderr else "Unknown FFmpeg error"
            raise Exception(f"FFmpeg encoding failed: {error_msg}")
    
    def _ffmpeg_encode(self, input_path, output_path, fps, codec, quality, format, audio, loop_count, is_sequence=False):
        """使用ffmpeg编码视频"""
        
        cmd = [ffmpeg_path]
        
        # 输入参数
        if is_sequence:
            cmd.extend(['-framerate', str(fps), '-i', input_path])
        else:
            cmd.extend(['-i', input_path])
        
        # 添加音频输入
        if audio is not None:
            audio_path = self._save_temp_audio(audio)
            cmd.extend(['-i', audio_path])
        
        # 编码参数
        if codec == "prores_ks":
            cmd.extend(['-c:v', 'prores_ks', '-profile:v', '4444'])
            if format == "mov":
                cmd.extend(['-pix_fmt', 'yuva444p10le'])  # 支持alpha
        elif codec == "png":
            cmd.extend(['-c:v', 'png'])
        elif codec == "qtrle":
            cmd.extend(['-c:v', 'qtrle'])
        
        if quality is not None and codec not in ["png", "qtrle"]:
            cmd.extend(['-crf', str(quality)])
        
        # 帧率
        cmd.extend(['-r', str(fps)])
        
        # 循环参数
        if loop_count > 0:
            cmd.extend(['-loop', str(loop_count)])
        
        # 音频参数
        if audio is not None:
            cmd.extend(['-c:a', 'aac'])
        
        # 输出
        cmd.extend(['-y', output_path])
        
        logger.info(f"FFmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            logger.info("FFmpeg encoding completed successfully")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode(*ENCODE_ARGS) if e.stderr else "Unknown FFmpeg error"
            raise Exception(f"FFmpeg encoding failed: {error_msg}")
    
    def _save_temp_audio(self, audio):
        """保存临时音频文件"""
        # 这里假设audio是一个包含音频数据的字典
        # 实际实现需要根据ComfyUI的音频格式进行调整
        temp_audio_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{uuid.uuid4().hex}.wav")
        # TODO: 实现音频保存逻辑
        return temp_audio_path


class KY_SaveImageSequence:
    """保存图像序列为PNG文件"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "KY_sequence"}),
                "format": (["png", "jpg", "tiff", "webp"], {"default": "png"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
            },
            "optional": {
                "mask": ("MASK",),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": BIGMAX}),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("output_dir", "frame_count")
    FUNCTION = "save_images"
    CATEGORY = "KY_Video"
    OUTPUT_NODE = True
    
    def save_images(self, images, filename_prefix, format, quality, mask=None, start_frame=0, end_frame=-1):
        """保存图像序列"""
        
        # 处理帧范围
        total_frames = len(images)
        if end_frame == -1:
            end_frame = total_frames
        else:
            end_frame = min(end_frame, total_frames)
        
        if start_frame >= end_frame:
            raise ValueError(f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}")
        
        # 选择帧范围
        selected_images = images[start_frame:end_frame]
        selected_mask = None
        if mask is not None:
            selected_mask = mask[start_frame:end_frame]
        
        # 创建输出目录
        output_dir = folder_paths.get_output_directory()
        timestamp = int(time.time())
        sequence_dir = os.path.join(output_dir, f"{filename_prefix}_{timestamp}")
        os.makedirs(sequence_dir, exist_ok=True)
        
        logger.info(f"Saving {len(selected_images)} images to {sequence_dir}")
        
        # 保存每一帧
        for i, img_tensor in enumerate(selected_images):
            frame_filename = f"{filename_prefix}_{i:06d}.{format}"
            frame_path = os.path.join(sequence_dir, frame_filename)
            
            # 转换张量到PIL图像
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='RGB')
            
            # 如果有mask且格式支持alpha，添加alpha通道
            if mask is not None and selected_mask is not None and format in ['png', 'tiff', 'webp']:
                alpha_np = (selected_mask[i].cpu().numpy() * 255).astype(np.uint8)
                img_pil = img_pil.convert('RGBA')
                img_pil.putalpha(Image.fromarray(alpha_np, mode='L'))
            
            # 保存图像
            if format in ['jpg', 'jpeg']:
                img_pil = img_pil.convert('RGB')  # JPEG不支持alpha
                img_pil.save(frame_path, format.upper(), quality=quality)
            elif format == 'png':
                img_pil.save(frame_path, "PNG")
            elif format == 'webp':
                img_pil.save(frame_path, "WEBP", quality=quality)
            elif format == 'tiff':
                img_pil.save(frame_path, "TIFF")
        
        logger.info(f"Saved {len(selected_images)} images successfully")
        return (sequence_dir, len(selected_images))


# 节点映射
NODE_CLASS_MAPPINGS = {
    "KY_SaveVideo": KY_SaveVideo,
    "KY_SaveImageSequence": KY_SaveImageSequence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KY_SaveVideo": "KY Save Video (Alpha Support)",
    "KY_SaveImageSequence": "KY Save Image Sequence",
}

