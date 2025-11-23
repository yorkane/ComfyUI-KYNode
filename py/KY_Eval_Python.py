import types

from server import PromptServer
import sys
import io
import traceback
import json, re, traceback

remove_type_name = re.compile(r"(\{.*\})", re.I | re.M)

# Hack: string type that is always equal in not equal comparisons, thanks pythongosssss
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


PY_CODE = AnyType("*")
IDEs_DICT = {}


# - Thank you very much for the class -> Trung0246 -
# - https://github.com/Trung0246/ComfyUI-0246/blob/main/utils.py#L51
class TautologyStr(str):
	def __ne__(self, other):
		return False


class ByPassTypeTuple(tuple):
	def __getitem__(self, index):
		if index > 0:
			index = 0
		item = super().__getitem__(index)
		if isinstance(item, str):
			return TautologyStr(item)
		return item
# ---------------------------


class KY_Eval_Python:
    def __init__(self): 
        self.js_complete = False
        self.js_result = None

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "pycode": (
                    "PYCODE",
                    {
                        "default": """import re, json, os, traceback
from time import strftime

def runCode():
    nowDataTime = strftime("%Y-%m-%d %H:%M:%S")
    return f"Hello ComfyUI with us today {nowDataTime}!"
r0_str = runCode()
""",
"multiline": True},
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ByPassTypeTuple((PY_CODE,))
    RETURN_NAMES =  ("r0_str",)
    FUNCTION = "exec_py"
    DESCRIPTION = "IDE Node is an node that allows you to run code written in Python or Javascript directly in the node."
    CATEGORY = "KYNode/Code"

    def exec_py(self, pycode, unique_id, extra_pnginfo, **kwargs): 
        if unique_id not in IDEs_DICT:
            IDEs_DICT[unique_id] = self


        outputs = {unique_id: unique_id}
        if extra_pnginfo and 'workflow' in extra_pnginfo and extra_pnginfo['workflow']:
            for node in extra_pnginfo['workflow']['nodes']:
                if node['id'] == int(unique_id):
                    outputs_valid = [ouput for ouput in node.get('outputs', []) if ouput.get('name','') != '' and ouput.get('type','') != '']
                    outputs = {ouput['name']: None for ouput in outputs_valid}
                    self.RETURN_TYPES = ByPassTypeTuple(out["type"] for out in outputs_valid)
                    self.RETURN_NAMES = tuple(name for name in outputs.keys())
        my_namespace = types.SimpleNamespace()
       # 从 prompt 对象中提取 prompt_id
        # if extra_data and 'extra_data' in extra_data and 'prompt_id' in extra_data['extra_data']:
        #     prompt_id = prompt['extra_data']['prompt_id']
        # outputs['p0_str'] = p0_str
            
        my_namespace.__dict__.update(outputs)            
        my_namespace.__dict__.update({prop: kwargs[prop] for prop in kwargs})
        # my_namespace.__dict__.setdefault("r0_str", "The r0 variable is not assigned")
        try:
            exec(pycode, my_namespace.__dict__)
        except Exception as e:
            err = traceback.format_exc()
            tb = traceback.extract_tb(sys.exc_info()[2])
            line_no = tb[-1].lineno if tb else 0
            PromptServer.instance.send_sync("python_editor_error", {
                "error": str(e),
                "line": line_no,
                "node_id": unique_id
            })
            print(f"Python执行错误 (节点ID: {unique_id}): 第{line_no}行 - {str(e)}")
            raise

        new_dict = {key: my_namespace.__dict__[key] for key in my_namespace.__dict__ if key not in ['__builtins__', *kwargs.keys()] and not callable(my_namespace.__dict__[key])}
        return (*new_dict.values(),)


NODE_CLASS_MAPPINGS = {
    "KY_Eval_Python": KY_Eval_Python
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KY_Eval_Python": "Eval Python Code"
}