import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 创建样式
const style = document.createElement("style");
style.textContent = `
    .ky-file-browser {
        display: flex;
        flex-direction: column;
        height: 100%;
        color: var(--fg-color);
        font-family: sans-serif;
    }
    .ky-browser-header {
        padding: 10px;
        background: var(--bg-color);
        border-bottom: 1px solid var(--border-color);
        display: flex;
        gap: 10px;
        align-items: center;
        flex-wrap: wrap;
    }
    .ky-filter-container {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .ky-filter-label {
        font-size: 12px;
        color: var(--fg-color);
    }
    .ky-filter-select {
        background: #333333;
        color: var(--input-text);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
        min-width: 120px;
    }
    .ky-current-path {
        flex-grow: 1;
        background: var(--input-bg);
        color: var(--input-text);
        padding: 5px;
        border-radius: 4px;
        border: 1px solid var(--border-color);
    }
    .ky-file-list {
        overflow-y: auto;
        padding: 10px;
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .ky-file-item {
        padding: 6px 10px;
        cursor: pointer;
        border-radius: 4px;
        display: flex;
        align-items: center;
    }
    .ky-file-item:hover {
        background: var(--tr-even-bg-color);
    }
    .ky-file-item.selected {
        background: var(--p-600, #3b82f6);
        color: #ffffff;
        border-left: 3px solid var(--p-800, #1e40af);
    }
    .ky-file-item.selected:hover {
        background: var(--p-600, #3b82f6);
    }
    .ky-item-icon {
        margin-right: 10px;
        width: 20px;
        text-align: center;
        display: inline-block;
    }
    .ky-browser-footer {
        padding: 10px;
        border-top: 1px solid var(--border-color);
        display: flex;
        justify-content: flex-end;
        gap: 10px;
    }
    .ky-btn {
        padding: 5px 15px;
        cursor: pointer;
        background: var(--comfy-input-bg);
        border: 1px solid var(--border-color);
        color: var(--fg-color);
        border-radius: 4px;
    }
    .ky-btn:hover {
        background: var(--comfy-menu-bg);
    }
    .ky-btn.primary {
        background: var(--p-700);
        color: white;
    }
    .ky-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    .ky-browser-body {
        display: flex;
        flex: 1;
        min-height: 0;
        border-top: 1px solid var(--border-color);
    }
    .ky-file-list {
        flex: 0 0 35%;
        border-right: 1px solid var(--border-color);
    }
    .ky-preview {
        flex: 0 0 65%;
        display: flex;
        flex-direction: column;
        padding: 10px;
        gap: 10px;
    }
    .ky-preview-title {
        font-size: 12px;
        color: var(--fg-color);
    }
    .ky-preview-content {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: auto;
        background: var(--tr-even-bg-color);
        border: 1px solid var(--border-color);
        border-radius: 4px;
    }
    .ky-preview-content img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
    .ky-header-meta {
        font-size: 12px;
        color: var(--fg-color);
        opacity: 0.8;
        white-space: nowrap;
    }
`;
document.head.appendChild(style);

// 根据文件扩展名获取图标
function getFileIcon(fileName) {
    if (!fileName || typeof fileName !== 'string') return "📄";
    
    const extension = fileName.split('.').pop().toLowerCase();
    
    // 图像文件
    const imageExtensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp', 'ico', 'tiff', 'tif'];
    if (imageExtensions.includes(extension)) return "🖼️";
    
    // 视频文件
    const videoExtensions = ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'm4v', '3gp', 'ogv'];
    if (videoExtensions.includes(extension)) return "🎬";
    
    // 音频文件
    const audioExtensions = ['mp3', 'wav', 'flac', 'aac', 'ogg', 'wma', 'm4a', 'opus'];
    if (audioExtensions.includes(extension)) return "🎵";
    
    // 文档文件
    const documentExtensions = ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'odt', 'ods', 'odp'];
    if (documentExtensions.includes(extension)) return "📋";
    
    // 代码文件
    const codeExtensions = ['js', 'jsx', 'ts', 'tsx', 'html', 'css', 'scss', 'less', 'json', 'xml', 'py', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go', 'rs', 'swift', 'kt', 'scala', 'r', 'sql', 'sh', 'bat', 'ps1'];
    if (codeExtensions.includes(extension)) return "💻";
    
    // 压缩文件
    const archiveExtensions = ['zip', 'rar', '7z', 'tar', 'gz', 'bz2', 'xz', 'lzma'];
    if (archiveExtensions.includes(extension)) return "📦";
    
    // 可执行文件
    const executableExtensions = ['exe', 'msi', 'app', 'deb', 'rpm', 'dmg', 'pkg'];
    if (executableExtensions.includes(extension)) return "⚙️";
    
    // 文本文件
    const textExtensions = ['txt', 'md', 'rtf', 'log', 'ini', 'cfg', 'conf', 'yaml', 'yml', 'toml'];
    if (textExtensions.includes(extension)) return "📝";
    
    // 电子表格文件
    const spreadsheetExtensions = ['csv', 'tsv'];
    if (spreadsheetExtensions.includes(extension)) return "📊";
    
    // 字体文件
    const fontExtensions = ['ttf', 'otf', 'woff', 'woff2', 'eot'];
    if (fontExtensions.includes(extension)) return "🔤";
    
    // 3D模型文件
    const modelExtensions = ['obj', 'fbx', 'dae', '3ds', 'blend', 'max', 'ma'];
    if (modelExtensions.includes(extension)) return "🎮";
    
    // 默认文件图标
    return "📄";
}

// 根据文件类型和过滤条件判断是否应该显示该文件
function shouldShowFile(file, filterType) {
    // 文件夹和驱动器始终显示，不受过滤条件影响
    if (file.type === "dir" || file.type === "drive") {
        return true;
    }
    
    // 如果是"folder"过滤条件，只显示文件夹
    if (filterType === "folder") {
        return false;
    }
    
    // 如果是"all"过滤条件，显示所有文件
    if (filterType === "all") {
        return true;
    }
    
    // 获取文件扩展名
    if (!file.name || typeof file.name !== 'string') return false;
    const extension = file.name.split('.').pop().toLowerCase();
    
    // 根据过滤类型判断
    switch (filterType) {
        case "image":
            return ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp', 'ico', 'tiff', 'tif'].includes(extension);
        case "video":
            return ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'm4v', '3gp', 'ogv'].includes(extension);
        case "audio":
            return ['mp3', 'wav', 'flac', 'aac', 'ogg', 'wma', 'm4a', 'opus'].includes(extension);
        case "document":
            return ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'odt', 'ods', 'odp'].includes(extension);
        case "code":
            return ['js', 'jsx', 'ts', 'tsx', 'html', 'css', 'scss', 'less', 'json', 'xml', 'py', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go', 'rs', 'swift', 'kt', 'scala', 'r', 'sql', 'sh', 'bat', 'ps1'].includes(extension);
        case "archive":
            return ['zip', 'rar', '7z', 'tar', 'gz', 'bz2', 'xz', 'lzma'].includes(extension);
        case "text":
            return ['txt', 'md', 'rtf', 'log', 'ini', 'cfg', 'conf', 'yaml', 'yml', 'toml', 'csv', 'tsv'].includes(extension);
        default:
            return true;
    }
}

app.registerExtension({
    name: "KY.PathSelector",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "KY_GetFromPath") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const dirWidget = this.widgets.find((w) => w.name === "path");

                this.addWidget("button", "📁 Open File Browser", null, async (widget, graphCanvas, node, pos, event) => {
                    const entered = (dirWidget?.value || "").trim().replace(/"/g, "");
                    if (!entered) {
                        showFileBrowser("output", (selectedPath) => { dirWidget.value = selectedPath; }, null, dirWidget);
                        return;
                    }
                    try {
                        const resp = await api.fetchApi("/ky_utils/check_path", { method: "POST", body: JSON.stringify({ path: entered }) });
                        const data = await resp.json();
                        if (data.type === "file") {
                            const parentDir = entered.substring(0, entered.lastIndexOf('\\')) || entered.substring(0, entered.lastIndexOf('/')) || entered;
                            showFileBrowser(parentDir, (selectedPath) => { dirWidget.value = selectedPath; }, entered, dirWidget);
                        } else if (data.type === "directory") {
                            showFileBrowser(entered, (selectedPath) => { dirWidget.value = selectedPath; }, null, dirWidget);
                        } else {
                            showFileBrowser("output", (selectedPath) => { dirWidget.value = selectedPath; }, null, dirWidget);
                            }
                    } catch (e) {
                        showFileBrowser(entered || "output", (selectedPath) => { dirWidget.value = selectedPath; }, null, dirWidget);
                    }
                });

                const originalCallback = dirWidget.callback;
                dirWidget.callback = function(value, ...args) {
                    if (originalCallback) {
                        originalCallback.call(this, value, ...args);
                    }
                };

                return r;
            };
        }
    },
});

// 处理路径输入的函数
async function handlePathInput(path, dirWidget) {
    try {
        // 规范化路径
        const normalizedPath = path.trim().replace(/"/g, '');
        
        // 检查路径是否存在
        const response = await api.fetchApi("/ky_utils/check_path", {
            method: "POST",
            body: JSON.stringify({ path: normalizedPath }),
        });
        
        const data = await response.json();
        
        if (data.error) {
            console.error("Path check error:", data.error);
            return;
        }
        
        // 如果是文件，打开文件浏览器并预览文件
        if (data.type === "file") {
            // 获取文件的父目录
            const parentDir = normalizedPath.substring(0, normalizedPath.lastIndexOf('\\')) || 
                             normalizedPath.substring(0, normalizedPath.lastIndexOf('/')) || 
                             normalizedPath;
            
            // 打开文件浏览器，显示父目录内容，并预览该文件
            showFileBrowser(parentDir, (selectedPath) => {
                // 更新路径输入框的值
                if (dirWidget) {
                    dirWidget.value = selectedPath;
                    // 不触发widget的回调，避免重新打开文件浏览器
                    // 用户已经通过文件浏览器选择了路径，不需要再次处理
                }
            }, normalizedPath, dirWidget); // 传递文件路径用于预览和dirWidget
        } 
        // 如果是目录，打开文件浏览器并显示目录内容
        else if (data.type === "directory") {
            showFileBrowser(normalizedPath, (selectedPath) => {
                // 更新路径输入框的值
                if (dirWidget) {
                    dirWidget.value = selectedPath;
                    // 不触发widget的回调，避免重新打开文件浏览器
                    // 用户已经通过文件浏览器选择了路径，不需要再次处理
                }
            }, null, dirWidget); // 传递dirWidget
        }
    } catch (error) {
        console.error("Error handling path input:", error);
    }
}

// 全局变量，跟踪当前打开的对话框
let currentDialog = null;
let suppressPathHandling = false;

function showFileBrowser(initialPath, onSelect, filePathToPreview = null, dirWidget = null) {
    // 如果已有对话框打开，先关闭它
    if (currentDialog && document.body.contains(currentDialog)) {
        document.body.removeChild(currentDialog);
        currentDialog = null;
    }
    
    const dialog = document.createElement("div");
    dialog.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.8); z-index: 10000;
        display: flex; justify-content: center; align-items: center;
    `;
    
    // 保存当前对话框引用
    currentDialog = dialog;

    const content = document.createElement("div");
    content.style.cssText = `
        width: 100vw; height: 100vh;
        background: var(--comfy-menu-bg);
        border-radius: 8px; border: 1px solid var(--border-color);
        display: flex; flex-direction: column; overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    `;
    
    dialog.appendChild(content);

    content.innerHTML = `
        <div class="ky-file-browser">
            <div class="ky-browser-header">
                <button class="ky-btn" id="ky-up-btn">⬆ Up</button>
                <div class="ky-filter-container">
                    <span class="ky-filter-label">Filter:</span>
                    <select class="ky-filter-select" id="ky-filter-select">
                        <option value="all">📄 All Files</option>
                        <option value="image">🖼️ Images</option>
                        <option value="video">🎬 Videos</option>
                        <option value="audio">🎵 Audio</option>
                        <option value="document">📋 Documents</option>
                        <option value="code">💻 Code</option>
                        <option value="archive">📦 Archives</option>
                        <option value="text">📝 Text</option>
                        <option value="folder">📁 Folders Only</option>
                    </select>
                </div>
                <input type="text" class="ky-current-path" id="ky-path-input" readonly />
                <span class="ky-header-meta" id="ky-header-meta"></span>
                <a class="ky-btn" id="ky-download-btn" style="display:none">⬇ Save</a>
            </div>
            <div class="ky-browser-body">
                <div class="ky-file-list" id="ky-file-list"></div>
                <div class="ky-preview" id="ky-preview">
                    <div class="ky-preview-title">Preview</div>
                    <div class="ky-preview-content" id="ky-preview-content"></div>
                </div>
            </div>
            <div class="ky-browser-footer">
                <button class="ky-btn" id="ky-cancel-btn">❌ Cancel</button>
                <button class="ky-btn primary" id="ky-select-btn">✅ Select</button>
            </div>
        </div>
    `;

    document.body.appendChild(dialog);

    const pathInput = content.querySelector("#ky-path-input");
    const fileListEl = content.querySelector("#ky-file-list");
    const upBtn = content.querySelector("#ky-up-btn");
    const cancelBtn = content.querySelector("#ky-cancel-btn");
    const selectBtn = content.querySelector("#ky-select-btn");
    const filterSelect = content.querySelector("#ky-filter-select");
    const previewEl = content.querySelector("#ky-preview");
    const previewContentEl = content.querySelector("#ky-preview-content");
    const headerMetaEl = content.querySelector("#ky-header-meta");
    const downloadBtn = content.querySelector("#ky-download-btn");

    let currentPath = initialPath || "";
    let parentPath = ""; // 由后端 API 提供
    let selectedItemPath = null;
    let currentFilter = "all"; // 当前过滤类型
    let allFiles = []; // 存储所有文件，用于过滤
    let initialFilePath = filePathToPreview; // 存储初始文件路径，用于预览
    let renderedFiles = [];
    let previewCache = new Map();
    let currentPreviewToken = 0;

    function finalizeSelection(finalPath) {
        if (finalPath === "My Computer") {
            return;
        }
        if (dirWidget) {
            suppressPathHandling = true;
            dirWidget.value = finalPath;
        }
        closeDialog();
        setTimeout(() => { suppressPathHandling = false; }, 0);
    }

    async function fetchPath(path) {
        try {
            const response = await api.fetchApi("/ky_utils/browse", {
                method: "POST",
                body: JSON.stringify({ path: path }),
            });
            const data = await response.json();
            
            if (data.error) {
                alert("Error: " + data.error);
                return;
            }

            render(data);
            
            // 如果有初始文件路径，在渲染完成后预览该文件
            if (initialFilePath) {
                // 查找文件列表中的文件
                const fileItem = allFiles.find(file => file.path === initialFilePath);
                if (fileItem) {
                    // 选中该文件并预览
                    selectFileAndPreview(fileItem);
                }
                initialFilePath = null; // 清除初始文件路径
            }
        } catch (e) {
            console.error(e);
            alert("Failed to browse path.");
        }
    }
    
    // 选中文件并预览
    function selectFileAndPreview(file) {
        const fileItems = document.querySelectorAll(".ky-file-item");
        for (const item of fileItems) {
            if (item.dataset && item.dataset.path === file.path) {
                document.querySelectorAll(".ky-file-item").forEach(i => i.classList.remove("selected"));
                item.classList.add("selected");
                selectedItemPath = file.path;
                pathInput.value = file.path;
                initialFilePath = null;
                updatePreview(file);
                if (item.scrollIntoView) item.scrollIntoView({ block: "nearest" });
                prefetchNeighbors();
                break;
            }
        }
    }

    function findFirstFileIndex() {
        const i = renderedFiles.findIndex(f => f.type === "file");
        return i === -1 ? 0 : i;
    }

    function findLastFileIndex() {
        for (let i = renderedFiles.length - 1; i >= 0; i--) {
            if (renderedFiles[i].type === "file") return i;
        }
        return renderedFiles.length - 1;
    }

    function findNextFileIndex(start, step) {
        let i = start + step;
        if (i < 0) i = 0;
        if (i > renderedFiles.length - 1) i = renderedFiles.length - 1;
        const dir = step >= 0 ? 1 : -1;
        while (i >= 0 && i < renderedFiles.length) {
            if (renderedFiles[i].type === "file") return i;
            i += dir;
        }
        return start;
    }

    function moveSelection(delta) {
        if (!renderedFiles || renderedFiles.length === 0) return;
        let idx = renderedFiles.findIndex(f => f.path === selectedItemPath);
        if (idx === -1) idx = findFirstFileIndex();
        const nextIdx = findNextFileIndex(idx, delta);
        selectFileAndPreview(renderedFiles[nextIdx]);
    }

    function getPageStep() {
        const firstItem = fileListEl.querySelector(".ky-file-item");
        const itemHeight = firstItem ? firstItem.offsetHeight : 24;
        const page = Math.floor(fileListEl.clientHeight / (itemHeight || 1));
        return page > 0 ? page : 10;
    }

    function prefetchNeighbors() {
        if (!renderedFiles || renderedFiles.length === 0 || !selectedItemPath) return;
        const idx = renderedFiles.findIndex(f => f.path === selectedItemPath);
        const neighbors = [idx - 1, idx + 1];
        for (const i of neighbors) {
            if (i >= 0 && i < renderedFiles.length) {
                const f = renderedFiles[i];
                if (f.type === "file") ensureCached(f);
            }
        }
    }

    async function ensureCached(file) {
        if (!file || file.type !== "file") return;
        if (previewCache.has(file.path)) return;
        try {
            const response = await api.fetchApi("/ky_utils/file_preview", {
                method: "POST",
                body: JSON.stringify({ path: file.path })
            });
            const info = await response.json();
            if (info && !info.error) {
                if (info.type === "image" && info.preview_url) {
                    const img = new Image();
                    const entry = { info, element: img };
                    previewCache.set(file.path, entry);
                    img.onload = () => {};
                    img.src = info.preview_url;
                } else if (info.type === "text") {
                    previewCache.set(file.path, { info });
                } else if ((info.type === "video" || info.type === "audio") && info.preview_url) {
                    previewCache.set(file.path, { info });
                } else {
                    previewCache.set(file.path, { info });
                }
            }
        } catch (e) {}
    }

    async function updatePreview(file) {
        const previewContentEl = document.querySelector("#ky-preview-content");
        if (!previewContentEl) return;
    if (!file || file.type !== "file") {
        if (headerMetaEl) headerMetaEl.textContent = "";
        if (downloadBtn) {
            downloadBtn.style.display = "none";
            downloadBtn.removeAttribute("href");
            downloadBtn.removeAttribute("download");
        }
        return;
    }
        const token = ++currentPreviewToken;
        let cached = previewCache.get(file.path);
        if (!cached) {
            try {
                const response = await api.fetchApi("/ky_utils/file_preview", {
                    method: "POST",
                    body: JSON.stringify({ path: file.path })
                });
                const info = await response.json();
                if (info.error) {
                if (token !== currentPreviewToken) return;
                if (headerMetaEl) headerMetaEl.textContent = "";
                return;
                }
                if (info.type === "image" && info.preview_url) {
                    const img = new Image();
                    cached = { info, element: img };
                    previewCache.set(file.path, cached);
                    img.onload = () => {
                        if (token !== currentPreviewToken) return;
                        clearPreview();
                        const clone = img.cloneNode();
                        previewContentEl.appendChild(clone);
                    };
                    img.src = info.preview_url;
                } else {
                    cached = { info };
                    previewCache.set(file.path, cached);
                }
            } catch (e) {
            if (token !== currentPreviewToken) return;
            previewContentEl.textContent = "Preview failed";
            return;
            }
        }
        const info = cached.info;
    const sizeStr = typeof info?.size === "number" ? `${info.size} bytes` : "";
    if (headerMetaEl) headerMetaEl.textContent = `${file.name}${sizeStr ? ` • ${sizeStr}` : ""}`;
    if (downloadBtn) {
        if (info?.preview_url) {
            downloadBtn.style.display = "";
            downloadBtn.setAttribute("href", info.preview_url);
            downloadBtn.setAttribute("download", file.name);
        } else {
            downloadBtn.style.display = "none";
            downloadBtn.removeAttribute("href");
            downloadBtn.removeAttribute("download");
        }
    }
        if (info?.type === "image" && cached.element) {
            if (cached.element.complete) {
                clearPreview();
                const clone = cached.element.cloneNode();
                previewContentEl.appendChild(clone);
            }
        } else if (info?.type === "text" && info.snippet) {
            clearPreview();
            const pre = document.createElement("pre");
            pre.style.whiteSpace = "pre-wrap";
            pre.style.wordBreak = "break-word";
            pre.textContent = info.snippet;
            previewContentEl.appendChild(pre);
        } else if (info?.type === "video" && info.preview_url) {
            clearPreview();
            const video = document.createElement("video");
            video.controls = true;
            video.style.width = "100%";
            video.style.height = "100%";
            video.src = info.preview_url;
            previewContentEl.appendChild(video);
        } else if (info?.type === "audio" && info.preview_url) {
            clearPreview();
            const audio = document.createElement("audio");
            audio.controls = true;
            audio.style.width = "100%";
            audio.src = info.preview_url;
            previewContentEl.appendChild(audio);
        } else {
            clearPreview();
            previewContentEl.textContent = "No preview available";
        }
        ensureCached(file);
        prefetchNeighbors();
    }

    function render(data) {
        // 更新状态
        currentPath = data.path;
        parentPath = data.parent_path; // 可能是路径，也可能是 "ROOT_DRIVES" 或空字符串
        
        // 更新 UI
        pathInput.value = currentPath;
        fileListEl.innerHTML = "";
        selectedItemPath = null;
        clearPreview();
        
        // Up 按钮状态：如果没有父级（且不是特殊的 ROOT_DRIVES 模式），则禁用
        upBtn.disabled = !parentPath;

        // 存储所有文件
        allFiles = data.files || [];
        
        // 应用过滤
        applyFilter();
    }
    
    function applyFilter() {
        fileListEl.innerHTML = "";
        selectedItemPath = null;
        clearPreview();
        
        // 根据当前过滤条件筛选文件
        const filteredFiles = allFiles.filter(file => shouldShowFile(file, currentFilter));
        renderedFiles = filteredFiles;
        
        filteredFiles.forEach((file, i) => {
            const el = document.createElement("div");
            el.className = "ky-file-item";
            
            // 根据类型显示不同图标
            let icon = "📄";
            if (file.type === "dir") icon = "📁";
            else if (file.type === "drive") icon = "💾"; // 硬盘图标
            else if (file.type === "file") icon = getFileIcon(file.name); // 根据文件扩展名获取图标
            
            el.innerHTML = `<span class="ky-item-icon">${icon}</span> ${file.name}`;
            el.dataset.path = file.path;
            el.dataset.index = String(i);
            
            el.onclick = () => {
                // 如果是文件夹或驱动器，点击进入
                // 如果是 ".." 也是进入
                const isNavigable = file.type === "dir" || file.type === "drive";
                
                if (isNavigable && file.name !== "..") {
                    fetchPath(file.path);
                } else if (file.name === "..") {
                    // 使用后端返回的 parent_path 会更稳，但点击列表中 .. 时通常 file.path 已经是正确父路径
                    fetchPath(file.path);
                } else {
                    // 选中文件
                    document.querySelectorAll(".ky-file-item").forEach(i => i.classList.remove("selected"));
                    el.classList.add("selected");
                    selectedItemPath = file.path;
                    pathInput.value = file.path;
                    // 清除初始文件路径，因为用户已经手动选择了文件
                    initialFilePath = null;
                    updatePreview(file);
                }
            };
            if (file.type === "file") {
                el.ondblclick = () => {
                    finalizeSelection(file.path);
                };
            }
            
            fileListEl.appendChild(el);
        });
    }

    // 事件绑定
    upBtn.onclick = () => {
        if (parentPath) {
            fetchPath(parentPath);
        }
    };

    cancelBtn.onclick = () => {
        closeDialog();
    };

    selectBtn.onclick = () => {
        const finalPath = initialFilePath || selectedItemPath || currentPath;
        if (finalPath === "My Computer") {
            alert("Please select a valid drive or folder.");
            return;
        }
        finalizeSelection(finalPath);
    };
    
    // 关闭对话框的函数
    function closeDialog() {
        if (currentDialog && document.body.contains(currentDialog)) {
            document.body.removeChild(currentDialog);
            currentDialog = null;
        }
    }

    // 过滤下拉框事件处理
    filterSelect.onchange = () => {
        currentFilter = filterSelect.value;
        applyFilter();
    };

    // 添加键盘事件监听器，只在对话框打开时有效
    const keyHandler = (e) => {
        // 确保事件只在对话框打开时处理
        if (!currentDialog || !document.body.contains(currentDialog)) {
            return;
        }
        
        if (e.key === "Escape") {
            // Esc键等同于点击取消按钮
            e.preventDefault();
            e.stopPropagation();
            cancelBtn.onclick();
        } else if (e.key === "Enter") {
            // Enter键等同于点击选择按钮
            e.preventDefault();
            e.stopPropagation();
            selectBtn.onclick();
        } else if (e.key === "ArrowDown") {
            e.preventDefault();
            e.stopPropagation();
            moveSelection(1);
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            e.stopPropagation();
            moveSelection(-1);
        } else if (e.key === "PageDown") {
            e.preventDefault();
            e.stopPropagation();
            moveSelection(getPageStep());
        } else if (e.key === "PageUp") {
            e.preventDefault();
            e.stopPropagation();
            moveSelection(-getPageStep());
        } else if (e.key === "Home") {
            e.preventDefault();
            e.stopPropagation();
            const idx = findFirstFileIndex();
            selectFileAndPreview(renderedFiles[idx]);
        } else if (e.key === "End") {
            e.preventDefault();
            e.stopPropagation();
            const idx = findLastFileIndex();
            selectFileAndPreview(renderedFiles[idx]);
        }
    };
    
    // 添加键盘事件监听器到对话框元素，而不是document
    dialog.addEventListener("keydown", keyHandler);
    
    // 确保对话框可以获得焦点
    dialog.tabIndex = -1;
    dialog.focus();

    // 初始化加载
    fetchPath(currentPath);
}

function clearPreview() {
    const previewContentEl = document.querySelector("#ky-preview-content");
    if (previewContentEl) previewContentEl.innerHTML = "";
}

//

//

//
