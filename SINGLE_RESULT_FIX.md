# 单个检测结果处理修复说明

## 问题描述
API返回了单个检测结果的字典格式，而不是预期的字典列表格式：

```
解析到的bbox数据: {'bbox_2d': [108, 184, 359, 845], 'label': 'yellow elephant'}
错误: bbox_data不是列表类型，是: dict
```

## 问题原因
不同的MiniCPM-V API实现可能返回不同的数据格式：
- **单个检测结果**: `{'bbox_2d': [x, y, w, h], 'label': 'object'}`
- **多个检测结果**: `[{'bbox_2d': [x, y, w, h], 'label': 'object1'}, {'bbox_2d': [x, y, w, h], 'label': 'object2'}]`

## 修复内容

### 兼容性处理
```python
# 处理单个检测结果（字典）或多个检测结果（字典列表）
if isinstance(bbox_data, dict):
    # 单个检测结果，转换为列表
    bbox_list = [bbox_data]
    debug_info += f"检测到单个结果，转换为列表处理\n"
elif isinstance(bbox_data, list):
    # 多个检测结果
    bbox_list = bbox_data
    debug_info += f"检测到{len(bbox_list)}个结果\n"
else:
    debug_info += f"错误: bbox_data类型不支持，是: {type(bbox_data).__name__}\n"
    return ([], None, None, None, json.dumps(bbox_data, ensure_ascii=False), debug_info)
```

### 统一处理逻辑
修复后的代码现在可以处理：

1. **单个检测结果**:
   ```json
   {'bbox_2d': [108, 184, 359, 845], 'label': 'yellow elephant'}
   ```

2. **多个检测结果**:
   ```json
   [
     {'bbox_2d': [108, 184, 359, 845], 'label': 'yellow elephant'},
     {'bbox_2d': [500, 300, 200, 400], 'label': 'blue car'}
   ]
   ```

## 预期的debug_info输出

修复后，你应该看到类似这样的输出：

```
原始尺寸: (1536, 1024)
解析到的bbox数据: {'bbox_2d': [108, 184, 359, 845], 'label': 'yellow elephant'}
检测到单个结果，转换为列表处理
处理第1个item: {'bbox_2d': [108, 184, 359, 845], 'label': 'yellow elephant'}
原始bbox: [108, 184, 359, 845]
还原bbox: [实际转换后的坐标]
```

## 测试方法

在ComfyUI中重新运行节点，现在应该能够正确处理这个单个检测结果，并输出正确的坐标转换信息。