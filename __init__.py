from .advanced_lying_sigma_sampler import AdvancedLyingSigmaSamplerNode
from .KY_SaveImageToPath import KY_SaveImageToPath

# 将节点映射到NODE_CLASS_MAPPINGS和NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "AdvancedLyingSigmaSampler": AdvancedLyingSigmaSamplerNode,
    "KY_SaveImageToPath": KY_SaveImageToPath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedLyingSigmaSampler": "Advanced Lying Sigma Sampler",
    "KY_SaveImageToPath": "Save Image To target Path"
}