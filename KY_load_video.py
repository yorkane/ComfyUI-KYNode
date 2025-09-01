import os
import itertools
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
import psutil
import subprocess
import re
import time

import folder_paths
from comfy.utils import common_upscale, ProgressBar
import nodes
from comfy.k_diffusion.utils import FolderOfImages
from .utils.logger import logger
from .utils.video_utils import BIGMAX, DIMMAX, calculate_file_hash, get_sorted_dir_files_from_directory,\
        lazy_get_audio, hash_path, validate_path, strip_path, try_download_video,  \
        is_url, imageOrLatent, ffmpeg_path, ENCODE_ARGS, floatOrInt


video_extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov']

VHSLoadFormats = {
    'None': {},
    'AnimateDiff': {'target_rate': 8, 'dim': (8,0,512,512)},
    'Mochi': {'target_rate': 24, 'dim': (16,0,848,480), 'frames':(6,1)},
    'LTXV': {'target_rate': 24, 'dim': (32,0,768,512), 'frames':(8,1)},
    'Hunyuan': {'target_rate': 24, 'dim': (16,0,848,480), 'frames':(4,1)},
    'Cosmos': {'target_rate': 24, 'dim': (16,0,1280,704), 'frames':(8,1)},
    'Wan': {'target_rate': 16, 'dim': (8,0,832,480), 'frames':(4,1)},
}

if not hasattr(nodes, 'VHSLoadFormats'):
    nodes.VHSLoadFormats = {}

def get_load_formats():
    #TODO: check if {**extra_config.VHSLoafFormats, **VHSLoadFormats} has minimum version
    formats = {}
    formats.update(nodes.VHSLoadFormats)
    formats.update(VHSLoadFormats)
    return (list(formats.keys()),
            {'default': 'AnimateDiff', 'formats': formats})
def get_format(format):
    if format in VHSLoadFormats:
        return VHSLoadFormats[format]
    return nodes.VHSLoadFormats.get(format, {})



def target_size(width, height, custom_width, custom_height, downscale_ratio=8) -> tuple[int, int]:
    if downscale_ratio is None:
        downscale_ratio = 8
    if custom_width == 0 and custom_height ==  0:
        pass
    elif custom_height == 0:
        height *= custom_width/width
        width = custom_width
    elif custom_width == 0:
        width *= custom_height/height
        height = custom_height
    else:
        width = custom_width
        height = custom_height
    width = int(width/downscale_ratio + 0.5) * downscale_ratio
    height = int(height/downscale_ratio + 0.5) * downscale_ratio
    return (width, height)

def ffmpeg_frame_generator(video, force_rate, frame_load_cap, start_time,
                           custom_width, custom_height, downscale_ratio=8,
                           apply_colorspace_fix=False, custom_colorspace="None", pix_fmt="rgba64le"):
    args_input = ["-i", video]
    args_dummy = [ffmpeg_path] + args_input +['-c', 'copy', '-frames:v', '1', "-f", "null", "-"]
    size_base = None
    fps_base = None
    if custom_colorspace == "None":
        apply_colorspace_fix = False
    else:
        apply_colorspace_fix = True
    try:
        dummy_res = subprocess.run(args_dummy, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception("An error occurred in the ffmpeg subprocess:\n" 
                + e.stderr.decode(*ENCODE_ARGS))
    lines = dummy_res.stderr.decode(*ENCODE_ARGS)
    if "Video: vp9 " in lines:
        args_input = ["-c:v", "libvpx-vp9"] + args_input
        args_dummy = [ffmpeg_path] + args_input +['-c', 'copy', '-frames:v', '1', "-f", "null", "-"]
        try:
            dummy_res = subprocess.run(args_dummy, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            raise Exception("An error occurred in the ffmpeg subprocess:\n" 
                    + e.stderr.decode(*ENCODE_ARGS))
        lines = dummy_res.stderr.decode(*ENCODE_ARGS)

    for line in lines.split('\n'):
        match = re.search("^ *Stream .* Video.*, ([1-9]|\\d{2,})x(\\d+)", line)
        if match is not None:
            size_base = [int(match.group(1)), int(match.group(2))]
            fps_match = re.search(", ([\\d\\.]+) fps", line)
            if fps_match:
                fps_base = float(fps_match.group(1))
            else:
                fps_base = 1
            alpha = re.search("(yuva|rgba|bgra)", line) is not None
            break
    else:
        raise Exception("Failed to parse video/image information. FFMPEG output:\n" + lines)

    durs_match = re.search("Duration: (\\d+:\\d+:\\d+\\.\\d+),", lines)
    if durs_match:
        durs = durs_match.group(1).split(':')
        duration = int(durs[0])*360 + int(durs[1])*60 + float(durs[2])
    else:
        duration = 0

    if start_time > 0:
        if start_time > 4:
            post_seek = ['-ss', '4']
            args_input = ['-ss', str(start_time - 4)] + args_input
        else:
            post_seek = ['-ss', str(start_time)]
    else:
        post_seek = []
    args_all_frames = [ffmpeg_path, "-v", "error", "-an"] + \
            args_input + ["-pix_fmt", pix_fmt] + post_seek

    vfilters = []
    # Add colorspace conversion filters if enabled
    if apply_colorspace_fix:
        vfilters.append(f"scale=in_color_matrix={custom_colorspace}:out_color_matrix={custom_colorspace}")
    if force_rate != 0:
        # Add fps filter after colorspace if both are enabled
        vfilters.append("fps=fps="+str(force_rate))
    if custom_width != 0 or custom_height != 0:
        size = target_size(size_base[0], size_base[1], custom_width,
                           custom_height, downscale_ratio=downscale_ratio)
        ar = float(size[0])/float(size[1])
        if abs(size_base[0]*ar-size_base[1]) >= 1:
            #Aspect ratio is changed. Crop to new aspect ratio before scale
            vfilters.append(f"crop=if(gt({ar}\\,a)\\,iw\\,ih*{ar}):if(gt({ar}\\,a)\\,iw/{ar}\\,ih)")
        size_arg = ':'.join(map(str,size))
        if apply_colorspace_fix:
            vfilters.append(f"scale={size_arg}:in_color_matrix={custom_colorspace}:out_color_matrix={custom_colorspace}")
        else:
            vfilters.append(f"scale={size_arg}")
    else:
        size = size_base
    if len(vfilters) > 0:
        args_all_frames += ["-vf", ",".join(vfilters)]
    yieldable_frames = (force_rate or fps_base)*duration
    if frame_load_cap > 0:
        args_all_frames += ["-frames:v", str(frame_load_cap)]
        yieldable_frames = min(yieldable_frames, frame_load_cap)
    yield (size_base[0], size_base[1], fps_base, duration, fps_base * duration,
           1/(force_rate or fps_base), yieldable_frames, size[0], size[1], alpha)
    args_all_frames += ["-f", "rawvideo", "-"]
    pbar = ProgressBar(yieldable_frames)
    
    # Determine bytes per image based on pixel format
    if pix_fmt == "rgb24":
        bytes_per_pixel = 3
    elif pix_fmt == "rgba64le":
        bytes_per_pixel = 8
    else:
        # Default to 4 bytes for most other formats (rgba, bgra, etc.)
        bytes_per_pixel = 4
        
    try:
        with subprocess.Popen(args_all_frames, stdout=subprocess.PIPE) as proc:
            #Manually buffer enough bytes for an image
            bpi = size[0] * size[1] * bytes_per_pixel
            current_bytes = bytearray(bpi)
            current_offset=0
            prev_frame = None
            while True:
                bytes_read = proc.stdout.read(bpi - current_offset)
                if bytes_read is None:#sleep to wait for more data
                    time.sleep(.1)
                    continue
                if len(bytes_read) == 0:#EOF
                    break
                current_bytes[current_offset:len(bytes_read)] = bytes_read
                current_offset+=len(bytes_read)
                if current_offset == bpi:
                    if prev_frame is not None:
                        yield prev_frame
                        pbar.update(1)
                    # Process frame based on pixel format
                    if pix_fmt == "rgb24":
                        prev_frame = np.frombuffer(current_bytes, dtype=np.uint8).reshape(size[1], size[0], 3) / 255.0
                    elif pix_fmt == "rgba64le":
                        prev_frame = np.frombuffer(current_bytes, dtype=np.dtype(np.uint16).newbyteorder("<")).reshape(size[1], size[0], 4) / (2**16-1)
                        if not alpha:
                            prev_frame = prev_frame[:, :, :-1]
                    else:
                        # Handle other formats as 4-channel by default
                        prev_frame = np.frombuffer(current_bytes, dtype=np.uint8).reshape(size[1], size[0], bytes_per_pixel) / 255.0
                        if not alpha and bytes_per_pixel == 4:
                            prev_frame = prev_frame[:, :, :-1]
                    current_offset = 0
    except BrokenPipeError as e:
        raise Exception("An error occured in the ffmpeg subprocess:\n" 
                + proc.stderr.read().decode(*ENCODE_ARGS))
    if prev_frame is not None:
        yield prev_frame

#Python 3.12 adds an itertools.batched, but it's easily replicated for legacy support
def batched(it, n):
    while batch := tuple(itertools.islice(it, n)):
        yield batch

def load_video(memory_limit_mb=None,
               generator=ffmpeg_frame_generator, format='None',  **kwargs):
    if 'force_size' in kwargs:
        kwargs.pop('force_size')
        logger.warn("force_size has been removed. Did you reload the webpage after updating?")
    format = get_format(format)
    kwargs['video'] = strip_path(kwargs['video'])
    downscale_ratio = format.get('dim', (1,))[0]
    
    gen = generator(downscale_ratio=downscale_ratio, **kwargs)
    (width, height, fps, duration, total_frames, target_frame_time, yieldable_frames, new_width, new_height, alpha) = next(gen)

    memory_limit = None
    if memory_limit_mb is not None:
        memory_limit *= 2 ** 20
    else:
        #TODO: verify if garbage collection should be performed here.
        #leaves ~128 MB unreserved for safety
        try:
            memory_limit = (psutil.virtual_memory().available + psutil.swap_memory().free) - 2 ** 27
        except:
            logger.warn("Failed to calculate available memory. Memory load limit has been disabled")
            memory_limit = BIGMAX
    #TODO: use better estimate for when vae is not None
    #Consider completely ignoring for load_latent case?
    max_loadable_frames = int(memory_limit//(width*height*3*(.1)))
    gen = itertools.islice(gen, max_loadable_frames)
    frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
    #Some minor wizardry to eliminate a copy and reduce max memory by a factor of ~2
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (new_height, new_width, 4 if alpha else 3)))))
    if len(images) == 0:
        raise RuntimeError("No frames generated")
    if 'frames' in format and len(images) % format['frames'][0] != format['frames'][1]:
        err_msg = f"The number of frames loaded {len(images)}, does not match the requirements of the currently selected format."
        if len(format['frames']) > 2 and format['frames'][2]:
            raise RuntimeError(err_msg)
        div, mod = format['frames'][:2]
        frames = (len(images) - mod) // div * div + mod
        images = images[:frames]
        #Commenting out log message since it's displayed in UI. consider further
        #logger.warn(err_msg + f" Output has been truncated to {len(images)} frames.")
    if 'start_time' in kwargs:
        start_time = kwargs['start_time']
    else:
        start_time = kwargs['skip_first_frames'] * target_frame_time
    target_frame_time *= kwargs.get('select_every_nth', 1)
    #Setup lambda for lazy audio capture
    audio = lazy_get_audio(kwargs['video'], start_time, kwargs['frame_load_cap']*target_frame_time)
    #Adjust target_frame_time for select_every_nth
    video_info = {
        "source_fps": fps,
        "source_frame_count": total_frames,
        "source_duration": duration,
        "source_width": width,
        "source_height": height,
        "loaded_fps": 1/target_frame_time,
        "loaded_frame_count": len(images),
        "loaded_duration": len(images) * target_frame_time,
        "loaded_width": new_width,
        "loaded_height": new_height,
    }
    return (images, len(images), audio, fps, video_info)

class LoadVideoByPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("STRING", {"placeholder": "X://insert/path/here.mp4", "vhs_path_extensions": video_extensions}),
                "force_rate": (floatOrInt, {"default": 0, "min": 0, "max": 60, "step": 1, "disable": 0}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1, "disable": 0}),
                "start_time": ("FLOAT", {"default": 0, "min": 0, "max": BIGMAX, "step": .001, "widgetType": "VHSTIMESTAMP"}),
            },
            "optional": {
                "Video_object": ("VIDEO", {"default": None}),
                "format": get_load_formats(),
                "color_space": (["None","bt709", "bt601", "smpte240m", "bt2020"], {"default": "None"}),
                "pix_fmt": (["rgba64le", "rgb24", "rgba", "bgra", "yuva420p", "yuva422p", "yuva444p"], {"default": "rgba64le"}),
            },
        }

    CATEGORY = "KY_Video"

    RETURN_TYPES = (imageOrLatent, "MASK", "AUDIO", "VHS_VIDEOINFO", "FLOAT", "STRING")
    RETURN_NAMES = ("IMAGE", "mask", "audio", "video_info", "fps", "video_path_url")

    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        video_object = kwargs.pop('Video_object', None)
        video_path = kwargs['video']
        # str is empty or trimmed empty
        if video_path == None or video_path.strip() == "":
            video_path = None
            if video_object is not None:
                video_path = getattr(video_object, '_VideoFromFile__file', None)
                kwargs['video'] = video_path
        if video_path is None or validate_path(video_path) != True:
            raise Exception("video is not a valid path: " + kwargs['video'])
        if is_url(video_path):
            kwargs['video'] = try_download_video(kwargs['video']) or kwargs['video']
        
        
        
        # Extract colorspace parameters
        custom_colorspace = kwargs.pop('color_space', "None")
        pix_fmt = kwargs.pop('pix_fmt', "rgba64le")
        
        
        image, _, audio, fps, video_info =  load_video(
            **kwargs, 
            generator=ffmpeg_frame_generator,
            custom_colorspace=custom_colorspace,
            pix_fmt=pix_fmt
        )
        if image.size(3) == 4:
            return (image[:,:,:,:3], 1-image[:,:,:,3], audio, video_info)
        return (image, torch.zeros(image.size(0), 64, 64, device="cpu"), audio, fps, video_info, video_path)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(s, video):
        return True
        # return validate_path(video, allow_none=True)

class CreateVideoObjectFromPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    CATEGORY = "KY_Video"

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video_object",)

    FUNCTION = "create_video_object"

    def create_video_object(self, video_path):
        # Validate the path
        if not video_path or not os.path.exists(video_path):
            raise Exception(f"Invalid video path: {video_path}")
            
        # Create a simple video object that stores the path
        class VideoFromFile:
            def __init__(self, file_path):
                self.__file = file_path
                
        video_object = VideoFromFile(video_path)
        return (video_object,)


VIDEO_LOAD_CLASS_MAPPINGS = {
    "KY_LoadVideoByPath": LoadVideoByPath,
    "KY_CreateVideoObjectFromPath": CreateVideoObjectFromPath,
}

VIDEO_LOAD_NAME_MAPPINGS = {
    "KY_LoadVideoByPath": "load video from path",
    "KY_CreateVideoObjectFromPath": "create video object from path",
}