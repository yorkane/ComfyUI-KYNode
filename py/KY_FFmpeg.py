import os
import subprocess
import shlex
import json

# ======================================================================================
# KY_FFmpeg core configuration and utility functions
# ======================================================================================

# Common GPU-accelerated encoder mapping (primarily NVIDIA, common in ComfyUI environments)
# Users pick a generic codec name; code selects the actual encoder based on GPU/CPU option
CODEC_MAP = {
    "H.264 (MP4)": {"cpu": "libx264", "gpu": "h264_nvenc", "ext": "mp4"},
    "H.265 (MP4)": {"cpu": "libx265", "gpu": "hevc_nvenc", "ext": "mp4"},
    "ProRes (MOV)": {"cpu": "prores_ks", "gpu": None, "ext": "mov"},
    "VP9 (WebM)": {"cpu": "libvpx-vp9", "gpu": None, "ext": "webm"},
}

# Common color space / transfer characteristics
COLORSPACE_MAP = {
    "BT.709 (HD)": "bt709",
    "BT.2020 (UHD/HDR)": "bt2020",
    "Unspecified": "unspecified",
}

COMMON_FPS = [2, 3, 6, 12.0, 15.0, 23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0, 120.0]
QUALITY_VALUE_CHOICES = ["12", "16", "18", "20", "22", "23", "28", "35", "40", "45", "50", "5M", "8M", "10M", "15M", "20M"]
ENCODER_CHOICES = ["libx264", "h264_nvenc", "libx265", "hevc_nvenc", "prores_ks", "libvpx-vp9"]
HW_ACCEL_CHOICES = ["", "-hwaccel cuda -vsync passthrough", "-hwaccel cuda", "-hwaccel vulkan", "-hwaccel dxva2"]

def _get_codec_params(codec_choice, quality_type, quality_value, gpu_accel):
    """Generate encoder and quality parameters based on user choices."""
    
    # 1) Determine encoder name (CPU vs GPU)
    codec_info = CODEC_MAP.get(codec_choice)
    if not codec_info:
        raise ValueError(f"Unknown codec choice: {codec_choice}")

    # Prefer GPU encoder when selected and available
    encoder = codec_info["cpu"]
    hw_accel_flags = []
    
    if gpu_accel and codec_info["gpu"]:
        # Common hardware acceleration flags for NVIDIA (NVENC)
        # Note: Different hardware/drivers may need different methods; use the most universal here.
        hw_accel_flags.extend(["-vsync", "passthrough", "-hwaccel", "cuda"]) 
        encoder = codec_info["gpu"]
        
    codec_flags = ["-c:v", encoder]

    # 2) Determine quality parameters (CRF/QP/Bitrate)
    quality_flags = []
    if quality_type == "CRF (Constant Rate Factor)":
        # CRF works for libx264/libx265/VP9 (CPU)
        quality_flags.extend(["-crf", str(quality_value)])
    elif quality_type == "QP (Quantization Parameter)":
        # QP works for NVENC (GPU)
        # QP is recommended for GPU encoders, typically between 1-51
        if "nvenc" in encoder:
            quality_flags.extend(["-qp", str(quality_value)])
        else:
            # If QP selected but not a GPU encoder, fall back to CRF
            quality_flags.extend(["-crf", str(quality_value)])
            print(f"WARN: QP is best for NVENC. Using CRF {quality_value} instead for {encoder}.")
    elif quality_type == "Bitrate (k/M)":
        # Bitrate works for any encoder
        quality_flags.extend(["-b:v", quality_value])
        
    return hw_accel_flags, codec_flags, quality_flags, codec_info["ext"]

def _get_colorspace_flags(colorspace_choice):
    """Generate color space flags for KY_FFmpeg."""
    colorspace_value = COLORSPACE_MAP.get(colorspace_choice, "unspecified")
    if colorspace_value == "unspecified":
        return []
    
    # Set color_primaries, color_trc, colorspace consistently
    return [
        "-color_primaries", colorspace_value, 
        "-color_trc", colorspace_value, 
        "-colorspace", colorspace_value
    ]

def _run_KY_FFmpeg(command_list):
    """Execute KY_FFmpeg command and handle errors."""
    command_str = " ".join(shlex.quote(c) for c in command_list)
    print(f"[KY_FFmpeg] Running command: {command_str}")
    
    # Use pipes to capture output and Popen to avoid blocking
    process = subprocess.Popen(
        command_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        encoding='utf-8',
        # Use multithreading to improve efficiency
        env=dict(os.environ, FF_THREAD_FLAGS="0x8") 
    )
    
    # Wait for the process to finish
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        error_message = f"KY_FFmpeg Error (Code {process.returncode}):\n{stderr}"
        print(f"ERROR: {error_message}")
        raise RuntimeError(error_message)
    
    print("[KY_FFmpeg] Command executed successfully.")
    # Optionally print partial output, especially progress (stdout) and detailed logs (stderr)
    # print(f"KY_FFmpeg STDOUT:\n{stdout[-500:]}")
    
    return {"status": "success", "command": command_str, "output": stdout}

# ======================================================================================
# 节点类定义
# ======================================================================================

class KY_FFmpegVideoToImages:
    """Node 1: Convert a video into an image sequence."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "C:/path/to/input.mp4", "multiline": False}),
                "output_dir": ("STRING", {"default": "C:/path/to/output/frames", "multiline": False}),
                "output_fps": (COMMON_FPS, {"default": 30.0}),
                "image_format": (["png", "jpg", "webp", "tiff"], {"default": "png"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_dir_path",)
    FUNCTION = "execute"
    CATEGORY = "Video/KY_FFmpeg"
    DESCRIPTION = "Converts a video to an image sequence (e.g., for frame editing). High-quality PNG is default."

    def execute(self, video_path, output_dir, output_fps, image_format):
        video_path = video_path.strip()
        output_dir = output_dir.strip()
        try:
            output_fps = float(output_fps)
        except Exception:
            output_fps = 30.0

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video file not found: {video_path}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # KY_FFmpeg command: -i input, -vf filter (set fps), %08d 8-digit sequence naming
        output_pattern = os.path.join(output_dir, f"%08d.{image_format}")
        
        pix_fmt = "rgb24"
        if image_format.lower() == "jpg":
            pix_fmt = "yuvj420p"
        elif image_format.lower() == "webp":
            pix_fmt = "yuv420p"

        command = [
            "ffmpeg", "-hide_banner",
            "-i", video_path,
            "-vf", f"fps={output_fps}",
            "-pix_fmt", pix_fmt,
            output_pattern
        ]
        
        _run_KY_FFmpeg(command)
        
        return (output_dir,)

class KY_FFmpegImagesToVideo:
    """Node 2: Encode an image sequence into a video."""
    
    @classmethod
    def INPUT_TYPES(s):
        codec_choices = list(CODEC_MAP.keys())
        colorspace_choices = list(COLORSPACE_MAP.keys())
        
        return {
            "required": {
                "input_dir": ("STRING", {"default": "C:/path/to/input/frames", "multiline": False}),
                "output_path": ("STRING", {"default": "C:/path/to/output.mp4", "multiline": False}),
                "input_fps": (COMMON_FPS, {"default": 30.0}),
                "codec_choice": (codec_choices, {"default": "H.264 (MP4)"}),
                "quality_type": (["CRF (Constant Rate Factor)", "QP (Quantization Parameter)", "Bitrate (k/M)"], {"default": "CRF (Constant Rate Factor)"}),
                "quality_value": (QUALITY_VALUE_CHOICES, {"default": "18"}),
                "colorspace_choice": (colorspace_choices, {"default": "BT.709 (HD)"}),
                "gpu_accel": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "execute"
    CATEGORY = "Video/KY_FFmpeg"
    DESCRIPTION = "Encodes an image sequence into a video file with advanced options."

    def execute(self, input_dir, output_path, input_fps, codec_choice, quality_type, quality_value, colorspace_choice, gpu_accel):
        input_dir = input_dir.strip()
        output_path = output_path.strip()
        try:
            input_fps = float(input_fps)
        except Exception:
            input_fps = 30.0

        # 1) Find the first frame to determine format
        # Assume all images share the same format and are named with %08d
        first_frame_path = None
        for ext in ["png", "jpg", "jpeg", "webp", "tiff"]:
            test_path = os.path.join(input_dir, f"00000001.{ext}")
            if os.path.exists(test_path):
                first_frame_path = os.path.join(input_dir, f"%08d.{ext}")
                break
        
        if not first_frame_path:
            raise FileNotFoundError(f"No image files found (e.g., 00000001.png) in: {input_dir}")
            
        # 2) Build encoder and quality parameters
        if quality_type == "Bitrate (k/M)":
            if not (str(quality_value).lower().endswith("m") or str(quality_value).lower().endswith("k")):
                quality_value = "10M"
        else:
            try:
                int(quality_value)
            except Exception:
                quality_value = "18" if quality_type == "CRF (Constant Rate Factor)" else "20"

        hw_accel_flags, codec_flags, quality_flags, default_ext = _get_codec_params(
            codec_choice, quality_type, quality_value, gpu_accel
        )
        colorspace_flags = _get_colorspace_flags(colorspace_choice)
        
        # 3) Construct output path (ensure extension matches container unless explicitly set)
        if not output_path.lower().endswith(f".{default_ext}"):
            output_path = f"{output_path.rsplit('.', 1)[0]}.{default_ext}"
            print(f"INFO: Output path adjusted to match container: {output_path}")

        # KY_FFmpeg command
        command = [
            "ffmpeg", "-hide_banner",
            # Hardware acceleration flags (must be set before inputs)
            *hw_accel_flags,
            # Input settings
            "-framerate", str(input_fps), 
            "-i", first_frame_path, 
            # Video encoding settings (H.264/H.265/ProRes etc.)
            *codec_flags,
            # Quality/compression parameters (CRF/QP/Bitrate)
            *quality_flags,
            # Color space parameters
            *colorspace_flags,
            # Pixel format (yuv420p is widely compatible)
            "-pix_fmt", "yuv420p",
            # Multithreading/speed optimization (important for CPU encoders)
            "-preset", "medium", 
            # Force overwrite output file
            "-y", 
            output_path
        ]

        _run_KY_FFmpeg(command)
        
        return (output_path,)

class KY_FFmpegAddAudio:
    """Node 3: Add an audio track to a video."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_in_path": ("STRING", {"default": "C:/path/to/video_only.mp4", "multiline": False}),
                "audio_in_path": ("STRING", {"default": "C:/path/to/audio_track.mp3", "multiline": False}),
                "output_path": ("STRING", {"default": "C:/path/to/final_video.mp4", "multiline": False}),
                "audio_codec": (["aac", "mp3", "copy"], {"default": "aac"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "execute"
    CATEGORY = "Video/KY_FFmpeg"
    DESCRIPTION = "Merges a video file and an audio file. Uses 'copy' for video stream for speed."

    def execute(self, video_in_path, audio_in_path, output_path, audio_codec):
        video_in_path = video_in_path.strip()
        audio_in_path = audio_in_path.strip()
        output_path = output_path.strip()
        
        if not os.path.exists(video_in_path):
            raise FileNotFoundError(f"Input video file not found: {video_in_path}")
        if not os.path.exists(audio_in_path):
            raise FileNotFoundError(f"Input audio file not found: {audio_in_path}")

        # KY_FFmpeg command: -i with two inputs, -c:v copy (no re-encode), -c:a audio codec
        # -map 0:v:0 (first video stream from first input), -map 1:a:0 (first audio stream from second input)
        # -shortest (stop at the shortest stream to match durations)
        
        command = [
            "ffmpeg", "-hide_banner",
            "-i", video_in_path, 
            "-i", audio_in_path, 
            "-c:v", "copy",
            "-c:a", audio_codec,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest", 
            "-y",
            output_path
        ]
        
        _run_KY_FFmpeg(command)
        
        return (output_path,)

class KY_FFmpegTrimVideo:
    """Node 4: Trim a video by start time and duration."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_in_path": ("STRING", {"default": "C:/path/to/input_video.mp4", "multiline": False}),
                "output_path": ("STRING", {"default": "C:/path/to/trimmed_video.mp4", "multiline": False}),
                "start_time_seconds": ("FLOAT", {"default": 10.0, "min": 0.0, "step": 0.1}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.1, "step": 0.1}),
                "copy_stream": ("BOOLEAN", {"default": True, "label": "Stream Copy (Fast, less precise start)"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "execute"
    CATEGORY = "Video/KY_FFmpeg"
    DESCRIPTION = "Trims a video using start time and duration. Stream Copy is fastest but start time might be slightly off."

    def execute(self, video_in_path, output_path, start_time_seconds, duration_seconds, copy_stream):
        video_in_path = video_in_path.strip()
        output_path = output_path.strip()

        if not os.path.exists(video_in_path):
            raise FileNotFoundError(f"Input video file not found: {video_in_path}")
        
        # Convert seconds to HH:MM:SS.ms format accepted by KY_FFmpeg
        def _to_hhmmss(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h:02d}:{m:02d}:{s:06.3f}"
        
        start_time_str = _to_hhmmss(start_time_seconds)
        duration_str = _to_hhmmss(duration_seconds)
        
        # Recommended fast trim: place -ss (seek) before -i (input)
        command = ["ffmpeg", "-hide_banner", "-ss", start_time_str, "-i", video_in_path]
        
        # -t sets duration
        command.extend(["-t", duration_str])
        
        # Stream copy or re-encode
        if copy_stream:
            # -c copy is fastest (no re-encode) but seeking with -ss is less precise
            command.extend(["-c", "copy"])
        else:
            # For precise trimming, re-encode (slower)
            # You can add encoder and quality params here; keep defaults for simplicity
            command.extend(["-c:v", "libx264", "-c:a", "aac"])

        command.extend(["-y", output_path])
        
        _run_KY_FFmpeg(command)
        
        return (output_path,)

class KY_FFmpegCustomCmd:
    """New node: allow users to execute a custom KY_FFmpeg command."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "command_template": ("STRING", {"default": "-i {input_1} -c:v {param_1} -crf 23 {output_file}", "multiline": True, "placeholder": "Use {input_1}, {output_file}, {param_1} etc. to reference parameters"}),
                "input_1": ("STRING", {"default": "C:/path/to/input.mp4", "multiline": False, "placeholder": "Input file path 1 ({input_1})"}),
                "output_file": ("STRING", {"default": "C:/path/to/custom_output.mp4", "multiline": False, "placeholder": "Output file path ({output_file})"}),
            },
            "optional": {
                "input_2": ("STRING", {"default": "", "multiline": False, "placeholder": "Input file path 2 ({input_2}, optional)"}),
                "param_1": (ENCODER_CHOICES, {"default": "libx264"}),
                "param_2": ("STRING", {"default": "", "multiline": False, "placeholder": "Custom parameter 2 ({param_2})"}),
                "param_3": ("STRING", {"default": "", "multiline": False, "placeholder": "Custom parameter 3 ({param_3})"}),
                "hw_accel_pre_flags": (HW_ACCEL_CHOICES, {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "execute"
    CATEGORY = "Video/KY_FFmpeg/Advanced"
    DESCRIPTION = "Executes a custom KY_FFmpeg command. Use {input_1}, {output_file}, {param_1}, etc., in the template."

    def execute(self, command_template, input_1, output_file, input_2="", param_1="", param_2="", param_3="", hw_accel_pre_flags=""):
        
        # 1) Build parameter dictionary
        params = {
            "input_1": input_1.strip(),
            "input_2": input_2.strip(),
            "output_file": output_file.strip(),
            "param_1": param_1.strip(),
            "param_2": param_2.strip(),
            "param_3": param_3.strip(),
        }
        
        # 2) Substitute parameters into the template
        try:
            # Use format() to substitute values
            # Note: using raw params; empty strings are fine and avoid KeyError for optional keys.
            custom_command_string = command_template.format(**params)
        except KeyError as e:
            raise ValueError(f"Command template contains an un-provided key: {e}. Available keys: {list(params.keys())}")
            
        # 3) Build final command list
        final_command_list = ["ffmpeg"]
        
        # Prepend hardware acceleration flags (e.g. -hwaccel cuda), typically before -i
        if hw_accel_pre_flags.strip():
            final_command_list.extend(shlex.split(hw_accel_pre_flags.strip()))

        # Append user-defined command section
        # shlex.split safely tokenizes command strings and handles quoted paths
        final_command_list.extend(shlex.split(custom_command_string))
        
        # 4) Run KY_FFmpeg (ensure -y to force overwrite and avoid prompts)
        if "-y" not in final_command_list:
             final_command_list.append("-y")
             
        _run_KY_FFmpeg(final_command_list)
        
        return (output_file,)


# ======================================================================================
# ComfyUI node registration
# ======================================================================================
NODE_CLASS_MAPPINGS = {
    "KY_FFmpegVideoToImages": KY_FFmpegVideoToImages,
    "KY_FFmpegImagesToVideo": KY_FFmpegImagesToVideo,
    "KY_FFmpegAddAudio": KY_FFmpegAddAudio,
    "KY_FFmpegTrimVideo": KY_FFmpegTrimVideo,
    "KY_FFmpegCustomCmd": KY_FFmpegCustomCmd, # 新增节点
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KY_FFmpegVideoToImages": "KY_FFmpeg: Video to Image Sequence",
    "KY_FFmpegImagesToVideo": "KY_FFmpeg: Image Sequence to Video",
    "KY_FFmpegAddAudio": "KY_FFmpeg: Merge Video/Audio",
    "KY_FFmpegTrimVideo": "KY_FFmpeg: Trim Video (Duration)",
    "KY_FFmpegCustomCmd": "KY_FFmpeg: Custom Command (Advanced)",
}
