import os
import folder_paths
import json
import torch
from PIL import Image
import numpy as np
import json

from nodes import SaveImage


class KY_SaveImageToPath(SaveImage):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"IMG": ("IMAGE",),
                     "path": ("STRING", {"default": "ComfyUI.png"})},
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save_image_to_path"
    OUTPUT_NODE = True
    CATEGORY = "KY Nodes/Image"

    def save_image_to_path(self, IMG, path="ComfyUI.png", prompt=None, extra_pnginfo=None):
        #results = self.save_images(images, filename_prefix, prompt, extra_pnginfo)
        saved_paths = []
        #folder_structure = []
        #folder_structure = json.loads(folder_structure)
        base_directory = path
        full_file_path = path
        images = IMG

        # Ensure base directory exists
        #os.makedirs(base_directory, exist_ok=True)

        for i, image in enumerate(images):
            # Convert the image tensor to a PIL Image
            img = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))

            # Create the full folder path based on the folder structure
            #full_folder_path = self.create_folder_path(base_directory, [])
            #os.makedirs(full_folder_path, exist_ok=True)

            # Create the file name and ensure it doesn't overwrite existing files
            #index = 0 + i
            #while True:
                #full_file_name = file_name_template.format(index=index)
            #    full_file_path = path #os.path.join(full_folder_path, full_file_name)
            #    if not os.path.exists(full_file_path):
            #        break
            #    index += 1

            # Save the image
            img.save(full_file_path)

            # Save metadata if provided
            #if metadata:
            #    metadata_file_name = f"{os.path.splitext(full_file_name)[0]}_metadata.txt"
            #    metadata_file_path = os.path.join(full_folder_path, metadata_file_name)
            #    with open(metadata_file_path, 'w') as f:
            #        f.write(metadata)

            saved_paths.append(path)
            break
        #return (", ".join(saved_paths),)
        #results = (", ".join(saved_paths))
        return {
            "ui": {
                #"images": results['ui']['IMG']
                "images": path
            },
            "result": (IMG,)
        }
    def create_folder_path(self, base_directory, folder_structure):
        path = base_directory
        for folder in folder_structure:
            path = os.path.join(path, folder['name'])
            for child in folder['children']:
                path = self.create_folder_path(path, [child])
        return path


