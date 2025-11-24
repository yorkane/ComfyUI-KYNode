import importlib.util
import importlib
import inspect
import json
from pathlib import Path
import textwrap
import sys

ROOT = Path(__file__).resolve().parent.parent
COMFY_ROOT = ROOT.parent.parent
PY_DIR = ROOT / "py"
WEB_DIR = ROOT / "web"
DOC_DIR = ROOT / "doc"
NODES_DIR = DOC_DIR / "nodes"
WEB_DOC_DIR = DOC_DIR / "web"

def ensure_dirs():
    DOC_DIR.mkdir(exist_ok=True)
    NODES_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DOC_DIR.mkdir(parents=True, exist_ok=True)

def ensure_py_package():
    init_file = PY_DIR / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))

def load_module(path: Path):
    ensure_py_package()
    mod_name = f"py.{path.stem}"
    try:
        return importlib.import_module(mod_name)
    except Exception:
        return None

def get_class_info(cls):
    src_file = inspect.getsourcefile(cls) or inspect.getfile(cls)
    lines, start = inspect.getsourcelines(cls)
    return src_file, start

def safe_call_input_types(cls):
    if hasattr(cls, "INPUT_TYPES"):
        try:
            res = cls.INPUT_TYPES()
            if isinstance(res, dict):
                return res
        except Exception:
            return {}
    return {}

def get_attr(cls, name, default=None):
    return getattr(cls, name, default)

def render_inputs(inputs_dict):
    parts = []
    required = inputs_dict.get("required", {})
    optional = inputs_dict.get("optional", {})
    if required:
        parts.append("- 必填参数:")
        for k, v in required.items():
            parts.append(render_param(k, v))
    if optional:
        parts.append("- 可选参数:")
        for k, v in optional.items():
            parts.append(render_param(k, v))
    return "\n".join(parts) if parts else "- 无"

def render_param(name, spec):
    t = None
    meta = {}
    if isinstance(spec, (tuple, list)) and spec:
        t = spec[0]
        if len(spec) > 1 and isinstance(spec[1], dict):
            meta = spec[1]
    elif isinstance(spec, str):
        t = spec
    opts = []
    if isinstance(meta.get("choices"), (list, tuple)):
        opts.append("选项:" + ", ".join(map(str, meta.get("choices"))))
    if "default" in meta:
        opts.append("默认:" + str(meta.get("default")))
    if meta.get("multiline"):
        opts.append("多行:true")
    if meta.get("forceInput"):
        opts.append("强制输入:true")
    if meta.get("min") is not None:
        opts.append("最小:" + str(meta.get("min")))
    if meta.get("max") is not None:
        opts.append("最大:" + str(meta.get("max")))
    opts_str = (" （" + "; ".join(opts) + ")") if opts else ""
    return f"  - `{name}`: `{t}`{opts_str}"

def render_returns(ret_types, is_list=False):
    if not ret_types:
        return "- 无"
    if isinstance(ret_types, (list, tuple)):
        types = ", ".join(map(lambda x: "`" + str(x) + "`", ret_types))
    else:
        types = "`" + str(ret_types) + "`"
    flag = "- 列表输出:true" if is_list else "- 列表输出:false"
    return f"- 返回类型: {types}\n{flag}"

def find_ui_keys(path: Path):
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    keys = set()
    for line in txt.splitlines():
        if "ui." in line:
            idx = line.find("ui.")
            seg = line[idx:].strip()
            k = seg.split()[0].strip('"\'\n,;:')
            keys.add(k)
    return sorted(keys)

def parse_static_mappings(txt):
    start = txt.find("NODE_CLASS_MAPPINGS")
    if start == -1:
        return {}
    brace = txt.find("{", start)
    end = txt.find("}", brace)
    block = txt[brace + 1:end] if brace != -1 and end != -1 else ""
    res = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().strip("'\"")
        v = v.strip().strip(", ")
        v = v.strip()
        v = v.replace(",", "")
        v = v.strip()
        if v:
            res[k] = v
    return res

def find_class_line(txt, cls_name):
    for i, line in enumerate(txt.splitlines(), 1):
        if line.strip().startswith("class ") and (cls_name in line):
            return i
    return None

def extract_attr_value(txt, cls_name, attr):
    found = False
    for i, line in enumerate(txt.splitlines(), 1):
        if line.strip().startswith("class ") and (cls_name in line):
            found = True
            continue
        if found:
            s = line.strip()
            if s.startswith(attr + " ") or s.startswith(attr + "="):
                val = s.split("=", 1)[1].strip()
                return val
            if s.startswith("class "):
                break
    return None

def extract_input_types_block(txt, cls_name):
    grabbed = []
    found_cls = False
    found_def = False
    for line in txt.splitlines():
        s = line
        if s.strip().startswith("class ") and (cls_name in s):
            found_cls = True
            continue
        if found_cls and s.strip().startswith("def INPUT_TYPES"):
            found_def = True
            grabbed.append(s)
            continue
        if found_def:
            grabbed.append(s)
            if s.strip().startswith("def ") or s.strip().startswith("class "):
                break
    return "\n".join(grabbed)

def write_module_doc(mod_path: Path, mod, out_path: Path):
    ui_keys = find_ui_keys(mod_path)
    lines = []
    title = f"# {mod_path.name}"
    lines.append(title)
    lines.append("")
    if mod is not None:
        display_map = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
        class_map = getattr(mod, "NODE_CLASS_MAPPINGS", {})
        for name, cls in class_map.items():
            disp = display_map.get(name, name)
            file_ref, start_line = get_class_info(cls)
            inputs_dict = safe_call_input_types(cls)
            returns = get_attr(cls, "RETURN_TYPES", None)
            is_list = bool(get_attr(cls, "OUTPUT_IS_LIST", False))
            func = get_attr(cls, "FUNCTION", None)
            cat = get_attr(cls, "CATEGORY", None)
            out_node = bool(get_attr(cls, "OUTPUT_NODE", False))
            lines.append(f"## {disp} ({name})")
            lines.append(f"- 所在文件: `{Path(file_ref).as_posix()}:{start_line}`")
            if cat:
                lines.append(f"- 分类: `{cat}`")
            if func:
                lines.append(f"- 方法: `{func}`")
            lines.append(f"- 输出节点: {'true' if out_node else 'false'}")
            lines.append("- 输入:")
            lines.append(render_inputs(inputs_dict))
            lines.append("- 输出:")
            lines.append(render_returns(returns, is_list))
            if ui_keys:
                lines.append("- UI 输出约定:")
                for k in ui_keys:
                    lines.append(f"  - `{k}`")
            if name.endswith("CustomCmd") and mod_path.name == "KY_FFmpeg.py":
                lines.append("- 命令模板占位符说明: 使用 `{变量}` 格式，如 `{input}`、`{output}`，通过 `format(**params)` 替换")
                lines.append("- 关键实现: `py/KY_FFmpeg.py:392` 使用模板格式化")
            lines.append("")
    else:
        txt = mod_path.read_text(encoding="utf-8", errors="ignore")
        mappings = parse_static_mappings(txt)
        for name, cls_name in mappings.items():
            lines.append(f"## {name}")
            line_no = find_class_line(txt, cls_name)
            if line_no:
                lines.append(f"- 所在文件: `{mod_path.as_posix()}:{line_no}`")
            cat = extract_attr_value(txt, cls_name, "CATEGORY")
            func = extract_attr_value(txt, cls_name, "FUNCTION")
            out_node = extract_attr_value(txt, cls_name, "OUTPUT_NODE")
            returns = extract_attr_value(txt, cls_name, "RETURN_TYPES")
            is_list = extract_attr_value(txt, cls_name, "OUTPUT_IS_LIST")
            if cat:
                lines.append(f"- 分类: `{cat}`")
            if func:
                lines.append(f"- 方法: `{func}`")
            if out_node:
                lines.append(f"- 输出节点: `{out_node}`")
            if returns:
                lines.append(f"- 返回类型: `{returns}`")
            if is_list:
                lines.append(f"- 列表输出: `{is_list}`")
            block = extract_input_types_block(txt, cls_name)
            lines.append("- 输入定义片段:")
            if block:
                lines.append("```python")
                lines.append(textwrap.dedent(block).strip())
                lines.append("```")
            else:
                lines.append("(未检测到 INPUT_TYPES 定义)")
            if ui_keys:
                lines.append("- UI 输出约定:")
                for k in ui_keys:
                    lines.append(f"  - `{k}`")
            if name.endswith("CustomCmd") and mod_path.name == "KY_FFmpeg.py":
                lines.append("- 命令模板占位符说明: 使用 `{变量}` 格式，如 `{input}`、`{output}`，通过 `format(**params)` 替换")
                lines.append("- 关键实现: `py/KY_FFmpeg.py:392` 使用模板格式化")
            lines.append("")
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")

def write_nodes_docs():
    for p in sorted(PY_DIR.glob("*.py")):
        mod = load_module(p)
        out = NODES_DIR / (p.stem + ".md")
        write_module_doc(p, mod, out)

def write_readme(mod_files):
    lines = []
    lines.append("# ComfyUI-KYNode 文档")
    lines.append("")
    lines.append("- 模块数: " + str(len(mod_files)))
    lines.append("- 文档位置: `doc/nodes/` 与 `doc/web/`")
    lines.append("")
    lines.append("## 节点模块索引")
    for p in mod_files:
        lines.append(f"- `{p.name}` → `doc/nodes/{p.stem}.md`")
    lines.append("")
    lines.append("## 前端与路由概述")
    lines.append("- 前端目录: `web/`，包含视频对比、Python 编辑器、文件路径工具等")
    lines.append("- 后端路由: 见 `doc/web/README.md`，包含 `ky_utils/*` 相关接口")
    DOC_DIR.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")

def extract_routes_from_files_py():
    p = PY_DIR / "KY_Files.py"
    if not p.exists():
        return []
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    routes = []
    for i, line in enumerate(txt.splitlines(), 1):
        if "/ky_utils/" in line:
            s = line.strip()
            routes.append((i, s))
    return routes

def write_web_readme():
    lines = []
    lines.append("# 前端与路由说明")
    lines.append("")
    lines.append("## 前端文件")
    files = []
    if WEB_DIR.exists():
        for p in sorted(WEB_DIR.glob("**/*")):
            if p.is_file():
                files.append(p.relative_to(ROOT).as_posix())
    for f in files:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## 后端路由 (来自 `py/KY_Files.py`) ")
    for ln, s in extract_routes_from_files_py():
        lines.append(f"- `py/KY_Files.py:{ln}`: `{s}`")
    WEB_DOC_DIR.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")

def enforce_line_limit():
    for p in DOC_DIR.rglob("*.md"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        n = len(txt.splitlines())
        if n > 500:
            short = "\n".join(txt.splitlines()[:500])
            p.write_text(short, encoding="utf-8")

def main():
    ensure_dirs()
    mod_files = sorted(PY_DIR.glob("*.py"))
    write_nodes_docs()
    write_web_readme()
    write_readme(mod_files)
    enforce_line_limit()

if __name__ == "__main__":
    main()
