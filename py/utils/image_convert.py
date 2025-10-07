import hashlib

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image


def mask2tensor(mask):
    image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    return image


def tensor2mask(image, channel='red'):
    channels = ['red', 'green', 'blue', 'alpha']
    mask = image[:, :, :, channels.index(channel)]
    return mask


def tensor2pil(image):
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def pil2hex(image):
    return hashlib.sha256(np.array(tensor2pil(image)).astype(np.uint16).tobytes()).hexdigest()


def pil2mask(image):
    image_np = np.array(image.convert('L')).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return mask


def mask2pil(mask):
    if mask.ndim > 2:
        mask = mask.squeeze(0)
    mask_np = mask.cpu().numpy().astype('uint8')
    mask_pil = Image.fromarray(mask_np, mode='L')
    return mask_pil


def pil2np(image):
    return np.array(image).astype(np.uint8)


def np2pil(image):
    return Image.fromarray(image)


def tensor2np(image):
    return np.array(T.ToPILImage()(image.permute(2, 0, 1)).convert('RGB'))


def np2tensor(image):
    return T.ToTensor()(image).permute(1, 2, 0)


def np2mask(image):
    return torch.from_numpy(image).unsqueeze(0)


def mask2np(image):
    return np.array(image.squeeze(0).cpu())


def image_posterize(image, threshold):
    image = image.mean(dim=3, keepdim=True)
    image = (image > threshold).float()
    image = image.repeat(1, 1, 1, 3)

    return image
