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
        background: var(--input-bg);
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
        flex-grow: 1;
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
        background: var(--p-700); 
        color: white;
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
        if (nodeType.comfyClass === "KY_GetPath") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const dirWidget = this.widgets.find((w) => w.name === "path");

                this.addWidget("button", "Open File Browser", null, (widget, graphCanvas, node, pos, event) => {
                    showFileBrowser(dirWidget.value, (selectedPath) => {
                        dirWidget.value = selectedPath;
                    });
                });

                return r;
            };
        }
    },
});

function showFileBrowser(initialPath, onSelect) {
    const dialog = document.createElement("div");
    dialog.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.8); z-index: 10000;
        display: flex; justify-content: center; align-items: center;
    `;

    const content = document.createElement("div");
    content.style.cssText = `
        width: 600px; height: 500px;
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
                        <option value="all">All Files</option>
                        <option value="image">Images</option>
                        <option value="video">Videos</option>
                        <option value="audio">Audio</option>
                        <option value="document">Documents</option>
                        <option value="code">Code</option>
                        <option value="archive">Archives</option>
                        <option value="text">Text</option>
                        <option value="folder">Folders Only</option>
                    </select>
                </div>
                <input type="text" class="ky-current-path" id="ky-path-input" readonly />
            </div>
            <div class="ky-file-list" id="ky-file-list"></div>
            <div class="ky-browser-footer">
                <button class="ky-btn" id="ky-cancel-btn">Cancel</button>
                <button class="ky-btn primary" id="ky-select-btn">Select Current Path</button>
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

    let currentPath = initialPath || "";
    let parentPath = ""; // 由后端 API 提供
    let selectedItemPath = null;
    let currentFilter = "all"; // 当前过滤类型
    let allFiles = []; // 存储所有文件，用于过滤

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
        } catch (e) {
            console.error(e);
            alert("Failed to browse path.");
        }
    }

    function render(data) {
        // 更新状态
        currentPath = data.path;
        parentPath = data.parent_path; // 可能是路径，也可能是 "ROOT_DRIVES" 或空字符串
        
        // 更新 UI
        pathInput.value = currentPath;
        fileListEl.innerHTML = "";
        selectedItemPath = null;
        
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
        
        // 根据当前过滤条件筛选文件
        const filteredFiles = allFiles.filter(file => shouldShowFile(file, currentFilter));
        
        filteredFiles.forEach(file => {
            const el = document.createElement("div");
            el.className = "ky-file-item";
            
            // 根据类型显示不同图标
            let icon = "📄";
            if (file.type === "dir") icon = "📁";
            else if (file.type === "drive") icon = "💾"; // 硬盘图标
            else if (file.type === "file") icon = getFileIcon(file.name); // 根据文件扩展名获取图标
            
            el.innerHTML = `<span class="ky-item-icon">${icon}</span> ${file.name}`;
            
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
                }
            };
            
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
        document.body.removeChild(dialog);
    };

    selectBtn.onclick = () => {
        const finalPath = selectedItemPath || currentPath;
        // 过滤掉 "My Computer" 这种虚拟路径
        if (finalPath === "My Computer") {
            alert("Please select a valid drive or folder.");
            return;
        }
        onSelect(finalPath);
        document.body.removeChild(dialog);
    };

    // 过滤下拉框事件处理
    filterSelect.onchange = () => {
        currentFilter = filterSelect.value;
        applyFilter();
    };

    // 初始化加载
    fetchPath(currentPath);
}