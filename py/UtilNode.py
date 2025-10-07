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
import logging
from comfy_execution.graph import ExecutionBlocker
        
_CATEGORY = "KYNode/Utils"


def is_integer(n):
    if n % 1 == 0:
        return True
    else:
        return False


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")

def safe_get_bbox(lst, index, default=None):
    if 0 <= index < len(lst):
        return [[lst[index]]]
    return default

def is_deep_empty(obj):
    if obj is None:
        return True
    if isinstance(obj, str):
        return obj.strip() == ''
    if isinstance(obj, torch.Tensor) and torch.all(obj == 0): # 判断是否为空 image 或者 mask
        print('判断是否为空 image 或者 mask sinstance(obj, torch.Tensor) and torch.all(obj == 0)')
        return True 
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
    if isinstance(obj, (int, float, complex)):
        return obj == 0
    return False  # 非空值（数字、非空字符串、对象等）

class KY_JoinToString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "prefix": ("STRING", {"default": ""}),
                "any1": (any_typ,),
                "any1_any2_gap": ("STRING", {"default": ""}),
                "any2": (any_typ,),
                "any2_any3_gap": ("STRING", {"default": ""}),
                "any3": (any_typ,),
                "suffix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = ("prefix+all-text+suffix", "prefix+any1+gap+suffix", "prefix+any2+gap+suffix")
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
        any1_any2_gap="",
        any2_any3_gap="",
        suffix="",
    ):
        stringified = ""

        if isinstance(any1, (int, float, bool, str)):
            stringified = str(any1)
        elif isinstance(any1, list):
            stringified = ", ".join(str(item) for item in any1)
        any1text = stringified
        any2text = ""
        if any1_any2_gap:  # Check if first_gap is not empty
            stringified = stringified + any1_any2_gap  # Add the first_gap

        if isinstance(any2, (int, float, bool, str)):
            stringified = stringified + str(any2)
            any2text = str(any2)
        elif isinstance(any2, list):
            stringified = stringified + ", ".join(str(item) for item in any2)
            any2text = ", ".join(str(item) for item in any2)

        if any2_any3_gap:  # Check if second_gap is not empty
            stringified = stringified + any2_any3_gap  # Add the second_gap

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
            prefix + any1text + any1_any2_gap + suffix,
            prefix + any2text + any2_any3_gap + suffix,
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
            match = re.search(regex, stringified, re.MULTILINE | re.DOTALL)
            if match:
                groups = match.groups()
                count = len(groups)
                if count >= 2:
                    return (match.group(group_number), match.group(1), match.group(2))
                elif count >= 1:
                    return (
                        match.group(group_number),
                        match.group(1),
                        "",
                    )
                elif 0 <= group_number <= len(groups):
                    return (match.group(group_number), "", "")
                else:
                    return ("", "", "")
            else:
                return ("", "", "")
        except re.error:
            return ("无效的正则表达式", "", "")


operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg,
    ast.Mod: op.mod,
    ast.BitAnd: op.and_,
    ast.BitOr: op.or_,
    ast.Invert: op.invert,
    ast.And: lambda a, b: 1 if a and b else 0,
    ast.Or: lambda a, b: 1 if a or b else 0,
    ast.Not: lambda a: 0 if a else 1,
    ast.RShift: op.rshift,
    ast.LShift: op.lshift,
}

# TODO: restructure args to provide more info, generate hint based on args to save duplication
functions = {
    "round": {
        "args": (1, 2),
        "call": lambda a, b=None: round(a, b),
        "hint": "number, dp? = 0",
    },
    "ceil": {"args": (1, 1), "call": lambda a: math.ceil(a), "hint": "number"},
    "floor": {"args": (1, 1), "call": lambda a: math.floor(a), "hint": "number"},
    "min": {"args": (2, None), "call": lambda *args: min(*args), "hint": "...numbers"},
    "max": {"args": (2, None), "call": lambda *args: max(*args), "hint": "...numbers"},
    "randomint": {
        "args": (2, 2),
        "call": lambda a, b: random.randint(a, b),
        "hint": "min, max",
    },
    "randomchoice": {
        "args": (2, None),
        "call": lambda *args: random.choice(args),
        "hint": "...numbers",
    },
    "sqrt": {"args": (1, 1), "call": lambda a: math.sqrt(a), "hint": "number"},
    "int": {"args": (1, 1), "call": lambda a=None: int(a), "hint": "number"},
    "iif": {
        "args": (3, 3),
        "call": lambda a, b, c=None: b if a else c,
        "hint": "value, truepart, falsepart",
    },
}


class KY_MathExpression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "d": (
                    "FLOAT",
                    {
                        "default": 0,
                        # "step": 0.5,
                    },
                ),
                "expression": (
                    "STRING",
                    {
                        "multiline": False,
                        "dynamicPrompts": False,
                    },
                ),
            },
            "optional": {
                "a": (any_typ,),
                "b": (any_typ,),
                "c": (any_typ,),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("int", "float", "text")
    FUNCTION = "evaluate"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = """
KY_MathExpression accept any value can be convert into numbers: 
- round(n), ceil(n), floor(n), int(n), sqrt(n)
- min(n1, n2, n3...),
- max(n1, n2, n3...)
- randomint(min_int_number, max_int_number):
- randomchoice(n1, n2, n4, n4...): random select arguments
- iif(bool-eval, is_true_val, is_false_val):  iif(1>2, c, d) = return d

"""

    @classmethod
    def IS_CHANGED(s, expression, **kwargs):
        if "random" in expression:
            return float("nan")
        return expression

    def get_widget_value(self, extra_pnginfo, prompt, node_name, widget_name):
        workflow = (
            extra_pnginfo["workflow"] if "workflow" in extra_pnginfo else {"nodes": []}
        )
        node_id = None
        for node in workflow["nodes"]:
            name = node["type"]
            if "properties" in node:
                if "Node name for S&R" in node["properties"]:
                    name = node["properties"]["Node name for S&R"]
            if name == node_name:
                node_id = node["id"]
                break
            if "title" in node:
                name = node["title"]
            if name == node_name:
                node_id = node["id"]
                break
        if node_id is not None:
            values = prompt[str(node_id)]
            if "inputs" in values:
                if widget_name in values["inputs"]:
                    value = values["inputs"][widget_name]
                    if isinstance(value, list):
                        raise ValueError(
                            "Converted widgets are not supported via named reference, use the inputs instead."
                        )
                    return value
            raise NameError(f"Widget not found: {node_name}.{widget_name}")
        raise NameError(f"Node not found: {node_name}.{widget_name}")

    def get_size(self, target, property):
        if isinstance(target, dict) and "samples" in target:
            # Latent
            if property == "width":
                return target["samples"].shape[3] * 8
            return target["samples"].shape[2] * 8
        else:
            # Image
            if property == "width":
                return target.shape[2]
            return target.shape[1]

    def evaluate(
        self, expression, prompt, extra_pnginfo={}, a=None, b=None, c=None, d=0
    ):
        expression = expression.replace("\n", " ").replace("\r", "")
        node = ast.parse(expression, mode="eval").body

        lookup = {"a": a, "b": b, "c": c, "d": d}

        def eval_op(node, l, r):
            l = eval_expr(l)
            r = eval_expr(r)
            l = l if isinstance(l, int) else float(l)
            r = r if isinstance(r, int) else float(r)
            return operators[type(node.op)](l, r)

        def eval_expr(node):
            if isinstance(node, ast.Constant) or isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                return eval_op(node, node.left, node.right)
            elif isinstance(node, ast.BoolOp):
                return eval_op(node, node.values[0], node.values[1])
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](eval_expr(node.operand))
            elif isinstance(node, ast.Attribute):
                if node.value.id in lookup:
                    if node.attr == "width" or node.attr == "height":
                        return self.get_size(lookup[node.value.id], node.attr)

                return self.get_widget_value(
                    extra_pnginfo, prompt, node.value.id, node.attr
                )
            elif isinstance(node, ast.Name):
                if node.id in lookup:
                    val = lookup[node.id]
                    if isinstance(val, (int, float, complex)):
                        return val
                    else:
                        raise TypeError(
                            f"Compex types (LATENT/IMAGE) need to reference their width/height, e.g. {node.id}.width"
                        )
                raise NameError(f"Name not found: {node.id}")
            elif isinstance(node, ast.Call):
                if node.func.id in functions:
                    fn = functions[node.func.id]
                    l = len(node.args)
                    if l < fn["args"][0] or (
                        fn["args"][1] is not None and l > fn["args"][1]
                    ):
                        if fn["args"][1] is None:
                            toErr = " or more"
                        else:
                            toErr = f" to {fn['args'][1]}"
                        raise SyntaxError(
                            f"Invalid function call: {node.func.id} requires {fn['args'][0]}{toErr} arguments"
                        )
                    args = []
                    for arg in node.args:
                        args.append(eval_expr(arg))
                    return fn["call"](*args)
                raise NameError(f"Invalid function call: {node.func.id}")
            elif isinstance(node, ast.Compare):
                l = eval_expr(node.left)
                r = eval_expr(node.comparators[0])
                if isinstance(node.ops[0], ast.Eq):
                    return 1 if l == r else 0
                if isinstance(node.ops[0], ast.NotEq):
                    return 1 if l != r else 0
                if isinstance(node.ops[0], ast.Gt):
                    return 1 if l > r else 0
                if isinstance(node.ops[0], ast.GtE):
                    return 1 if l >= r else 0
                if isinstance(node.ops[0], ast.Lt):
                    return 1 if l < r else 0
                if isinstance(node.ops[0], ast.LtE):
                    return 1 if l <= r else 0
                raise NotImplementedError(
                    "Operator " + node.ops[0].__class__.__name__ + " not supported."
                )
            else:
                raise TypeError(node)

        r = eval_expr(node)
        if is_integer(r):
            s = str(int(r))
        else:
            s = str(r)

        return {
            "ui": {"value": [r]},
            "result": (
                int(r),
                float(r),
                s,
            ),
        }

class KY_AnyByIndex:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "ANY": (any_typ, {"forceInput": True}),
                "index": ("INT", {"forceInput": False, "default": 0}),
            }
        }
    
    TITLE = "Get item from list"
    RETURN_TYPES = (any_typ, )
    INPUT_IS_LIST = True
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    DESCRIPTION = """
    Get item from any list or batch by index
    - For single item (int/float/bool/str): return itself
    - For list (str/int/float/bool): return item by index
    - For image batch: return single image (keep batch dimension)
    - For image list: return single image (keep batch dimension)
    """

    def run(self, ANY, index):
        idx = index[0]

        # 处理输入列表
        if isinstance(ANY, list):
            # 如果是空列表，直接返回
            if not ANY:
                return (None,)
            
            # 获取第一个元素（因为 INPUT_IS_LIST = True，ANY 总是列表）
            first_item = ANY[0]
            
            # 如果第一个元素是列表（说明输入是列表的列表）
            if isinstance(first_item, list):
                if idx >= len(first_item):
                    raise ValueError(f"Index {idx} out of range for list size {len(first_item)}")
                return (first_item[idx],)
            if isinstance(first_item, tuple):
                return (first_item[idx],)
            
            # 如果第一个元素是 tensor
            if isinstance(first_item, torch.Tensor):
                if len(first_item.shape) == 4:  # 图像列表
                    # 检查是否为图像列表的列表
                    if isinstance(ANY, list) and len(ANY) > 1 and all(isinstance(x, torch.Tensor) and len(x.shape) == 4 for x in ANY):
                        if idx >= len(ANY):
                            raise ValueError(f"Index {idx} out of range for image list size {len(ANY)}")
                        return (ANY[idx],)
                    # 单个图像的情况
                    return (first_item[idx:idx+1],)
            
            # 如果是基本类型列表
            if isinstance(first_item, (int, float, bool, str)):
                if idx >= len(ANY):
                    raise ValueError(f"Index {idx} out of range for list size {len(ANY)}")
                return (ANY[idx],)
            
            return (first_item,)
        
        # 处理基本类型
        if isinstance(ANY, (int, float, bool, str)):
            return (ANY,)
            
        # 其他情况直接返回
        return (ANY,)

class KY_AnyToList:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "any1": (any_typ, {"forceInput": True}),
            },
            "optional": {
                "any2": (any_typ,),
                "any3": (any_typ,),
            }
        }
    
    TITLE = "Anything to List"
    RETURN_TYPES = (any_typ, "INT")
    RETURN_NAMES = ("merged_list", "total_items")
    OUTPUT_IS_LIST = (True, False)
    INPUT_IS_LIST = True
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    DESCRIPTION = """
    Merge up to 3 inputs into a single list:
    - For basic types (int/float/bool/str): add as single item
    - For lists: extend the result list
    - For image batch: convert to list of single images
    - For tensor batch: convert to list of single tensors
    Returns:
    - merged_list: Combined list of all inputs
    - total_items: Total number of items in the merged list
    """

    def process_input(self, input_value):
        if input_value is None:
            return []
            
        # 因为INPUT_IS_LIST=True，input_value总是列表
        result = []
        for item in input_value:
            result.extend(self.process_single_input(item))
        return result

    def process_single_input(self, value):
        # 如果是列表，直接返回
        if isinstance(value, list):
            return value
            
        # 如果是元组，转换为列表
        if isinstance(value, tuple):
            return list(value)
            
        # 处理图像或张量批次
        if isinstance(value, torch.Tensor):
            if len(value.shape) == 4:  # 图像批次
                return [value[i:i+1] for i in range(value.shape[0])]
            elif len(value.shape) > 1:  # 其他张量批次
                return [value[i] for i in range(value.shape[0])]
            else:  # 单个张量
                return [value]
                
        # 处理基本类型
        if isinstance(value, (int, float, bool, str)):
            return [value]
            
        # 其他类型作为单个元素处理
        return [value]

    def run(self, any1, any2=None, any3=None):
        result = []
        
        # 处理所有输入
        result.extend(self.process_input(any1))
        result.extend(self.process_input(any2))
        result.extend(self.process_input(any3))
        
        
        # 因为INPUT_IS_LIST=True，我们需要返回单个元素的列表
        return ([result], len(result))

class KY_isNone_blocker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_typ,)
            },
            "optional": {
                "reverse": ("BOOLEAN", {"default": False}),
                "block_execution": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = (any_typ, "BOOLEAN")
    RETURN_NAMES = ("out", "is_empty",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, any, reverse, block_execution):
        is_empty = is_deep_empty(any)
        if block_execution and is_empty and not reverse:
            return (ExecutionBlocker(None), is_empty)
        if reverse:
            return (any, not is_empty,)
        else:
            return (any, is_empty,)

class KY_First_NOT_EMPTY:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Fallback": (any_typ,)
            },
            "optional": {
                "A": (any_typ,),
                "B": (any_typ,),
                "C": (any_typ,),
                "D": (any_typ,),
                "force_fallback": ("BOOLEAN", {"default": True}),
                "block_if_all_empty": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = (any_typ, "BOOLEAN")
    RETURN_NAMES = ("out", "is_blocked",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = """
    Return first not NONE value/Image/Mask :
    - Empty list, Dictionary, Turple
    - Empty String
    - 0 float int complex
    - empty tensor
    """

    def execute(self, A = None, B = None, C = None, D = None, Fallback = None, force_fallback = False, block_if_all_empty = False):
        if force_fallback:
            if is_deep_empty(Fallback) and block_if_all_empty:
                return (ExecutionBlocker(None), True)
            return (Fallback, False)
        val = None
        if not is_deep_empty(A):
            val = A
        elif not is_deep_empty(B):
            val = B
        elif not is_deep_empty(C):
            val = C
        elif not is_deep_empty(D):
            val = D
        else:
            val = Fallback
        bol = is_deep_empty(val)

        # return (kwargs['in'] if kwargs['continue'] else ExecutionBlocker(None),)
        if block_if_all_empty and bol:
            return (ExecutionBlocker(None), True)
        else:
            return (val, False,)

class KY_MergeToJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input_1": ("STRING", {"multiline": True}),
                "json_input_2": ("STRING", {"multiline": True}),
                "merge_strategy": (["override", "preserve", "concat"], {"default": "override"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_json",)
    FUNCTION = "merge_json"
    CATEGORY = _CATEGORY

    def merge_json(self, json_input_1: str, json_input_2: str, merge_strategy: str) -> tuple[str]:
        try:
            data1 = json.loads(json_input_1)
            data2 = json.loads(json_input_2)
            
            if isinstance(data1, list) and isinstance(data2, list):
                result = data1 + data2
            elif isinstance(data1, dict) and isinstance(data2, dict):
                result = self._merge_dicts(data1, data2, merge_strategy)
            else:
                raise ValueError("Both inputs must be of the same type (either objects or arrays)")
                
            return (json.dumps(result, indent=2),)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON input")

    def _merge_dicts(self, dict1: dict, dict2: dict, strategy: str) -> dict:
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key not in result:
                result[key] = value
            else:
                if strategy == "override":
                    result[key] = value
                elif strategy == "preserve":
                    continue
                elif isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_dicts(result[key], value, strategy)
                elif isinstance(result[key], list) and isinstance(value, list):
                    result[key] = result[key] + value
                    
        return result


NODE_CLASS_MAPPINGS = {
    "KY_JoinToString": KY_JoinToString,
    "KY_RegexReplace": KY_RegexReplace,
    "KY_RegexExtractor": KY_RegexExtractor,
    "KY_MathExpression": KY_MathExpression,
    "KY_AnyByIndex": KY_AnyByIndex,
    "KY_AnyToList": KY_AnyToList,
    "KY_isNone": KY_isNone_blocker,
    "KY_MergeToJSON":  KY_MergeToJSON,
    "KY_First_NOT_EMPTY": KY_First_NOT_EMPTY,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KY_JoinToString": "Join any into string",
    "KY_RegexReplace": "Replace text by regex",
    "KY_RegexExtractor": "Extract text by regex",
    "KY_MathExpression": "Math expression eval",
    "KY_AnyByIndex": "Anything Get By Index",
    "KY_AnyToList": "Anything To List",
    "KY_isNone": "Block if None or empty ",
    "KY_MergeToJSON":  "Merge Json",
    "KY_First_NOT_EMPTY": "First NOT NONE",
}
