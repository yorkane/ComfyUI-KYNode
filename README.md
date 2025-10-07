# ComfyUI-KYNode
For comfyui custom nodes utils

## 功能节点

### 图像处理
- Load images from URL/Path/Base64
![kynode-1](https://github.com/user-attachments/assets/a8425f17-3772-457e-b2fc-91efb448c409)

### 视频处理
- Video Compare: 对比两个视频（支持URL和视频文件）并在预览窗口中显示
  - 支持两种输入模式：URL和视频文件对象
  - 可以通过拖拽中间的分隔线来对比两个视频
  - 提供滑块控制对比比例
  - 可以在预览窗口中动态更新视频URL
  - 自动处理ComfyUI的视频文件路径（input/output文件夹）

- Video File Compare: 直接从input/output文件夹中选择视频文件进行对比
  - 下拉菜单选择input或output文件夹中的视频文件
  - 无需连接其他节点，直接预览对比

- KY Load Video: 仿造ComfyUI官方Load Video节点
  - 从input或output文件夹加载视频文件
  - 下拉菜单选择视频文件
  - 输出视频对象和预览URL

- KY Save Video: 保存图像序列为支持alpha通道的视频文件
  - 支持将IMAGE张量序列保存为MOV/AVI/MP4格式视频
  - 支持alpha通道（通过MASK输入）
  - 多种编码器选择：ProRes 4444、PNG、QuickTime RLE
  - 可调节视频质量、帧率等参数
  - 支持添加音频轨道
  - 自动生成预览URL

- KY Save Image Sequence: 保存图像序列为独立的图片文件
  - 支持PNG、JPG、TIFF、WebP格式
  - 支持alpha通道保存（PNG/TIFF/WebP）
  - 可设置图片质量和帧范围
