import base64
import io
import os
import re

import numpy as np
from PIL import Image

_CATEGORY = "KYNode/Text"


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")


class KY_JoinToString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "any1": (any_typ,),
                "any2": (any_typ,),
                "any3": (any_typ,),
                "prefix": ("STRING", {"default": ""}),
                "first_gap": ("STRING", {"default": ""}),
                "second_gap": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = ("all-text", "any1-text", "any2-text")
    FUNCTION = "stringifyAny"
    CATEGORY = _CATEGORY
    DESCRIPTION = """
Converts any type to a string.
"""

    def stringifyAny(
        self,
        any1="",
        any2="",
        any3="",
        prefix="",
        first_gap="",
        second_gap="",
        suffix="",
    ):
        stringified = ""

        if isinstance(any1, (int, float, bool, str)):
            stringified = str(any1)
        elif isinstance(any1, list):
            stringified = ", ".join(str(item) for item in any1)
        any1text = stringified
        any2text = ""
        if first_gap:  # Check if first_gap is not empty
            stringified = stringified + first_gap  # Add the first_gap

        if isinstance(any2, (int, float, bool, str)):
            stringified = stringified + str(any2)
            any2text = str(any2)
        elif isinstance(any2, list):
            stringified = stringified + ", ".join(str(item) for item in any2)
            any2text = ", ".join(str(item) for item in any2)

        if second_gap:  # Check if second_gap is not empty
            stringified = stringified + second_gap  # Add the second_gap

        if isinstance(any3, (int, float, bool, str)):
            stringified = stringified + str(any3)
        elif isinstance(any3, list):
            stringified = stringified + ", ".join(str(item) for item in any3)

        if prefix:  # Check if prefix is not empty
            stringified = prefix + stringified  # Add the prefix
        if suffix:  # Check if suffix is not empty
            stringified = stringified + suffix  # Add the suffix
        return (
            stringified,
            any1text,
            any2text,
        )


class KY_RegexReplace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "regex1": ("STRING", {"default": ""}),
                "replace1": ("STRING", {"default": ""}),
            },
            "optional": {
                "any1": (any_typ,),
                "any2": (any_typ,),
                "regex2": ("STRING", {"default": ""}),
                "replace2": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "regex1-result",
        "final-result",
    )
    FUNCTION = "replace"
    CATEGORY = _CATEGORY
    DESCRIPTION = """
Converts any type to a string.
"""

    def replace(
        self,
        any1="",
        any2="",
        regex1="",
        replace1="",
        input_string="",
        regex2="",
        replace2="",
    ):
        stringified = ""
        if isinstance(any1, (int, float, bool, str)):
            stringified = str(any1)
        elif isinstance(any1, list):
            stringified = ", ".join(str(item) for item in any1)

        if isinstance(any2, (int, float, bool, str)):
            stringified = stringified + str(any2)
        elif isinstance(any2, list):
            stringified = stringified + ", ".join(str(item) for item in any2)

        if input_string:  # Check if prefix is not empty
            stringified = stringified + input_string  # Add the prefix

        result1 = re.sub(regex1, replace1, stringified)
        result2 = ""
        if regex2:
            result2 = re.sub(regex2, replace2, result1)
        return (
            result1,
            result2,
        )


class KY_RegexExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "regex": ("STRING", {"default": ""}),
                "group_number": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
            "optional": {
                "any1": (any_typ,),
                "any2": (any_typ,),
                "input_string": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "group_N",
        "group_1",
        "group_2",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "使用正则表达式从输入字符串中提取文本"

    def execute(self, input_string="", regex="", group_number=1, any1="", any2=""):
        stringified = ""
        if isinstance(any1, (int, float, bool, str)):
            stringified = str(any1)
        elif isinstance(any1, list):
            stringified = ", ".join(str(item) for item in any1)

        if isinstance(any2, (int, float, bool, str)):
            stringified = stringified + str(any2)
        elif isinstance(any2, list):
            stringified = stringified + ", ".join(str(item) for item in any2)
        stringified = stringified + input_string
        try:
            match = re.search(regex, stringified)
            if match:
                groups = match.groups()
                count = len(groups)
                if count >= 1:
                    return (
                        match.group(group_number),
                        match.group(1),
                    )
                if count >= 2:
                    return (match.group(group_number), match.group(1), match.group(2))
                if 0 <= group_number <= len(groups):
                    return (match.group(group_number),)
                else:
                    raise Exception("Regex don't have match group: " + group_number)
                    return ("",)
            else:
                raise Exception("Regex don't match: " + stringified)
                return ("",)
        except re.error:
            return ("无效的正则表达式",)


UTIL_NODE_CLASS_MAPPINGS = {
    "KY_JoinToString": KY_JoinToString,
    "KY_RegexReplace": KY_RegexReplace,
    "KY_RegexExtractor": KY_RegexExtractor,
}

UTIL_NODE_NAME_MAPPINGS = {
    "KY_JoinToString": "Join any into string",
    "KY_RegexReplace": "Replace text by regex",
    "KY_RegexExtractor": "Extract text by regex",
}
