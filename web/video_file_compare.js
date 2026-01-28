// 用于实现视频对比的JavaScript代码
import { app } from "/scripts/app.js";

// 注册节点扩展
app.registerExtension({
    name: "KYNode.VideoCompare",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "KY_VideoCompare") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                ensureVCStyle();
                this.compareWidget = this.addWidget("button", "🔍 Compare", null, () => {
                    const savedA = this._vcLastA;
                    const savedB = this._vcLastB;
                    const a = (savedA !== undefined && savedA !== null && savedA !== "")
                        ? savedA
                        : (this.widgets.find(w => w.name === "video_a_url_or_filepath")?.value || "");
                    const b = (savedB !== undefined && savedB !== null && savedB !== "")
                        ? savedB
                        : (this.widgets.find(w => w.name === "video_b_url_or_filepath")?.value || "");
                    openCompareDialog(a, b);
                });
                this.clearWidget = this.addWidget("button", "🧹 Clear", null, () => {
                    const wa = this.widgets.find(w => w.name === "video_a_url_or_filepath");
                    const wb = this.widgets.find(w => w.name === "video_b_url_or_filepath");
                    if (wa) wa.value = "";
                    if (wb) wb.value = "";
                });
                return r;
            };
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message && typeof message === "object") {
                    let a = message.video_a_source || "";
                    let b = message.video_b_source || "";
                    if (Array.isArray(a)) a = a.join("");
                    if (Array.isArray(b)) b = b.join("");
                    this._vcLastA = a;
                    this._vcLastB = b;
                    const wa = this.widgets.find(w => w.name === "video_a_url_or_filepath");
                    const wb = this.widgets.find(w => w.name === "video_b_url_or_filepath");
                    if (wa && !isTempPreview(a)) wa.value = a;
                    if (wb && !isTempPreview(b)) wb.value = b;
                    if (vcDialog) updateCompareDialog(a, b);
                }
            };
        }
        if (nodeData.name === "KY_ImageCompare") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                ensureVCStyle();
                this.compareWidget = this.addWidget("button", "🔍 Compare", null, () => {
                    const savedA = this._icLastA;
                    const savedB = this._icLastB;
                    const a = (savedA !== undefined && savedA !== null && savedA !== "")
                        ? savedA
                        : (this.widgets.find(w => w.name === "image_a_url_or_filepath")?.value || "");
                    const b = (savedB !== undefined && savedB !== null && savedB !== "")
                        ? savedB
                        : (this.widgets.find(w => w.name === "image_b_url_or_filepath")?.value || "");
                    openImageCompareDialog(a, b);
                });
                this.clearWidget = this.addWidget("button", "🧹 Clear", null, () => {
                    const wa = this.widgets.find(w => w.name === "image_a_url_or_filepath");
                    const wb = this.widgets.find(w => w.name === "image_b_url_or_filepath");
                    if (wa) wa.value = "";
                    if (wb) wb.value = "";
                });
                return r;
            };
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message && typeof message === "object") {
                    let a = message.image_a_source || "";
                    let b = message.image_b_source || "";
                    if (Array.isArray(a)) a = a.join("");
                    if (Array.isArray(b)) b = b.join("");
                    this._icLastA = a;
                    this._icLastB = b;
                    const wa = this.widgets.find(w => w.name === "image_a_url_or_filepath");
                    const wb = this.widgets.find(w => w.name === "image_b_url_or_filepath");
                    if (wa && !isTempPreview(a)) wa.value = a;
                    if (wb && !isTempPreview(b)) wb.value = b;
                    if (vcDialog) updateImageCompareDialog(a, b);
                }
            };
        }
    }
});

// 打开视频对比窗口的函数
let vcStyleInjected = false;
let vcDialog = null;
function isTempPreview(u){ if(!u || typeof u !== 'string') return false; return (u.includes('/api/view?') && (u.includes('subfolder=ky_compare') || u.includes('type=temp'))); }

function ensureVCStyle() {
    if (vcStyleInjected) return;
    const style = document.createElement("style");
    style.id = "ky-video-compare-style";
    style.textContent = `
    .ky-vc-overlay{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8);z-index:10000;display:flex;justify-content:center;align-items:center}
    .ky-vc-content{width:100vw;height:100vh;background:var(--comfy-menu-bg);border-radius:0;border:1px solid var(--border-color);display:flex;flex-direction:column;overflow:hidden}
    .ky-vc-content.ky-vc-fullscreen{border:none}
    .ky-vc-header{padding:8px;background:var(--bg-color);border-bottom:1px solid var(--border-color);display:flex;gap:10px;align-items:center}
    .ky-vc-content.ky-vc-fullscreen .ky-vc-header{display:none}
    .ky-vc-controls{display:flex;gap:8px;flex:1}
    .ky-vc-input{flex:1;background:var(--input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px;padding:4px 8px;font-size:12px}
    .ky-vc-btn{padding:5px 12px;background:var(--comfy-input-bg);border:1px solid var(--border-color);color:var(--fg-color);border-radius:4px;cursor:pointer}
    .ky-vc-btn.primary{background:var(--p-700);color:#fff}
    .ky-vc-body{flex:1;padding:0}
    #ky-vc-container{position:relative;width:100%;height:100%;line-height:0;background:#000;overflow:hidden}
    .ky-vc-zoom-layer{position:absolute;top:0;left:0;width:100%;height:100%;transform-origin:0 0;will-change:transform}
    #ky-vc-container>div>video{position:absolute;top:0;left:0;background:#000}
    #ky-vc-container>div>img{position:absolute;top:0;left:0;background:#000}
    #ky-vc-clipper{width:50%;position:absolute;top:0;bottom:0;overflow:hidden;border-right:1px dashed #ccc}
    #ky-vc-clipper video{position:absolute;top:0;left:0;background:#000}
    #ky-vc-clipper img{position:absolute;top:0;left:0;background:#000}
    `;
    document.head.appendChild(style);
    vcStyleInjected = true;
}

function openCompareDialog(a, b) {
    ensureVCStyle();
    closeCompareDialog();
    vcDialog = document.createElement("div");
    vcDialog.className = "ky-vc-overlay";
    const content = document.createElement("div");
    content.className = "ky-vc-content";
    content.innerHTML = `
        <div class="ky-vc-header">
            <button class="ky-vc-btn" id="ky-vc-close">✖ Close</button>
            <div class="ky-vc-controls">
                <button class="ky-vc-btn primary" id="ky-vc-reload">🔄 Reload</button>
                <button class="ky-vc-btn" id="ky-vc-fs">⛶ FullScreen</button>
                <input class="ky-vc-input" id="ky-vc-a" placeholder="Video A URL">
                <input class="ky-vc-input" id="ky-vc-b" placeholder="Video B URL">
            </div>
        </div>
        <div class="ky-vc-body">
            <div id="ky-vc-container">
                <div class="ky-vc-zoom-layer">
                    <video loop autoplay muted>
                        <source id="ky-vc-a-src">
                    </video>
                    <div id="ky-vc-clipper">
                        <video loop autoplay muted style="width:200%;z-index:3;">
                            <source id="ky-vc-b-src">
                        </video>
                    </div>
                </div>
            </div>
        </div>
    `;
    vcDialog.appendChild(content);
    document.body.appendChild(vcDialog);
    const inpA = content.querySelector('#ky-vc-a');
    const inpB = content.querySelector('#ky-vc-b');
    const aSrc = content.querySelector('#ky-vc-a-src');
    const bSrc = content.querySelector('#ky-vc-b-src');
    const vContainer = content.querySelector('#ky-vc-container');
    const clipper = content.querySelector('#ky-vc-clipper');
    const vA = vContainer.getElementsByTagName('video')[0];
    const vB = clipper.getElementsByTagName('video')[0];
    const zoomLayer = vContainer.querySelector('.ky-vc-zoom-layer');

    function toStr(x){return (Array.isArray(x)?x.join(''):x||"").toString();}
    inpA.value = toStr(a);
    inpB.value = toStr(b);
    aSrc.src = toStr(a);
    bSrc.src = toStr(b);
    vA.load();
    vB.load();

    let scale = 1, panX = 0, panY = 0, isDragging = false, lastX = 0, lastY = 0;
    function updateTransform() { zoomLayer.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`; }

    function track(e){
        if (isDragging) return;
        const rect=zoomLayer.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const pos=(clientX-rect.left)/rect.width*100;
        if(pos>=0 && pos<=100){clipper.style.width=pos+"%";vB.style.zIndex=3;}
    }

    function layout(){
        const cw = vContainer.clientWidth; const ch = vContainer.clientHeight;
        const aw = vA.videoWidth||1920; const ah = vA.videoHeight||1080;
        const ar = aw/ah; let tw, th;
        if (cw/ch > ar) { th = ch; tw = Math.round(ch*ar); } else { tw = cw; th = Math.round(cw/ar); }
        const left = Math.round((cw - tw)/2); const top = Math.round((ch - th)/2);
        Object.assign(vA.style,{width:tw+"px",height:th+"px",left:left+"px",top:top+"px"});
        Object.assign(vB.style,{width:tw+"px",height:th+"px",left:left+"px",top:top+"px"});
    }
    function onMeta(){layout();}
    vA.addEventListener('loadedmetadata', onMeta);
    vB.addEventListener('loadedmetadata', onMeta);
    window.addEventListener('resize', layout);

    vContainer.addEventListener('mousemove', (e) => {
        if (isDragging) {
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;
            panX += dx;
            panY += dy;
            updateTransform();
        } else {
            track(e);
        }
    });
    vContainer.addEventListener('mousedown', (e) => {
        if (e.button === 2) {
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
            e.preventDefault();
        }
    });
    vContainer.addEventListener('mouseup', () => { isDragging = false; });
    vContainer.addEventListener('contextmenu', e => e.preventDefault());
    vContainer.addEventListener('wheel', (e) => {
        e.preventDefault();
        const zoomSpeed = 0.1;
        const oldScale = scale;
        if (e.deltaY < 0) scale *= (1 + zoomSpeed);
        else scale /= (1 + zoomSpeed);
        scale = Math.max(0.1, Math.min(scale, 10));
        
        const rect = vContainer.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const rx = (mx - panX) / oldScale;
        const ry = (my - panY) / oldScale;
        panX = mx - rx * scale;
        panY = my - ry * scale;
        updateTransform();
    }, { passive: false });

    // Touch support for slider (basic)
    vContainer.addEventListener('touchstart',track,false);
    vContainer.addEventListener('touchmove',track,false);
    function sync(){if(vA.paused&&vB.paused)return;const d=Math.abs(vA.currentTime-vB.currentTime);if(d>0.1){vB.currentTime=vA.currentTime;}}
    const syncTimer=setInterval(sync,100);
    content.querySelector('#ky-vc-reload').addEventListener('click',()=>{aSrc.src=inpA.value;bSrc.src=inpB.value;vA.load();vB.load();});
    const closeBtn=content.querySelector('#ky-vc-close');
    const fsBtn=content.querySelector('#ky-vc-fs');
    function close(){clearInterval(syncTimer);closeCompareDialog();}
    closeBtn.addEventListener('click',close);
    vcDialog.addEventListener('click',(e)=>{if(e.target===vcDialog) close();});
    fsBtn.addEventListener('click',()=>{
        const el = content;
        if (!document.fullscreenElement) {
            el.requestFullscreen?.();
            el.classList.add('ky-vc-fullscreen');
        } else {
            document.exitFullscreen?.();
            el.classList.remove('ky-vc-fullscreen');
        }
    });
    document.addEventListener('fullscreenchange', ()=>{
        const isFs = document.fullscreenElement === content;
        content.classList.toggle('ky-vc-fullscreen', isFs);
        layout();
    });
    vcDialog.tabIndex=-1;vcDialog.addEventListener('keydown',(e)=>{
        if(e.key==='Escape'){
            e.preventDefault();
            if(document.fullscreenElement){
                document.exitFullscreen?.();
                return;
            }
            close();
        } else if (e.key.toLowerCase()==='f') {
            e.preventDefault();
            if (!document.fullscreenElement) { content.requestFullscreen?.(); content.classList.add('ky-vc-fullscreen'); }
            else { document.exitFullscreen?.(); content.classList.remove('ky-vc-fullscreen'); }
        }
    });
    vcDialog.focus();
    layout();
}

function updateCompareDialog(a,b){
    if(!vcDialog) return openCompareDialog(a,b);
    const content = vcDialog.querySelector('.ky-vc-content');
    const inpA = content.querySelector('#ky-vc-a');
    const inpB = content.querySelector('#ky-vc-b');
    const aSrc = content.querySelector('#ky-vc-a-src');
    const bSrc = content.querySelector('#ky-vc-b-src');
    function toStr(x){return (Array.isArray(x)?x.join(''):x||"").toString();}
    const sa = toStr(a); const sb = toStr(b);
    inpA.value = sa; inpB.value = sb;
    aSrc.src = sa; bSrc.src = sb;
    const vContainer = content.querySelector('#ky-vc-container');
    const clipper = content.querySelector('#ky-vc-clipper');
    const vA = vContainer.getElementsByTagName('video')[0];
    const vB = clipper.getElementsByTagName('video')[0];
    vA.load(); vB.load();
}

function closeCompareDialog(){
    if(vcDialog && document.body.contains(vcDialog)){
        document.body.removeChild(vcDialog);
        vcDialog=null;
    }
}

function openImageCompareDialog(a, b){
    ensureVCStyle();
    closeCompareDialog();
    vcDialog = document.createElement("div");
    vcDialog.className = "ky-vc-overlay";
    const content = document.createElement("div");
    content.className = "ky-vc-content";
    content.innerHTML = `
        <div class="ky-vc-header">
            <button class="ky-vc-btn" id="ky-vc-close">✖ Close</button>
            <div class="ky-vc-controls">
                <button class="ky-vc-btn primary" id="ky-vc-reload">🔄 Reload</button>
                <button class="ky-vc-btn" id="ky-vc-fs">⛶ FullScreen</button>
                <input class="ky-vc-input" id="ky-vc-a" placeholder="Image A URL">
                <input class="ky-vc-input" id="ky-vc-b" placeholder="Image B URL">
            </div>
        </div>
        <div class="ky-vc-body">
            <div id="ky-vc-container">
                <div class="ky-vc-zoom-layer">
                    <img id="ky-ic-a">
                    <div id="ky-vc-clipper">
                        <img id="ky-ic-b" style="z-index:3;">
                    </div>
                </div>
            </div>
        </div>
    `;
    vcDialog.appendChild(content);
    document.body.appendChild(vcDialog);
    const inpA = content.querySelector('#ky-vc-a');
    const inpB = content.querySelector('#ky-vc-b');
    const vContainer = content.querySelector('#ky-vc-container');
    const clipper = content.querySelector('#ky-vc-clipper');
    const imgA = content.querySelector('#ky-ic-a');
    const imgB = content.querySelector('#ky-ic-b');
    function toStr(x){return (Array.isArray(x)?x.join(''):x||"").toString();}
    inpA.value = toStr(a);
    inpB.value = toStr(b);
    imgA.src = toStr(a);
    imgB.src = toStr(b);

    let scale = 1, panX = 0, panY = 0, isDragging = false, lastX = 0, lastY = 0;
    const zoomLayer = vContainer.querySelector('.ky-vc-zoom-layer');
    function updateTransform() { zoomLayer.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`; }

    function track(e){
        if(isDragging) return;
        const rect=zoomLayer.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const pos=(clientX-rect.left)/rect.width*100;
        if(pos>=0 && pos<=100){clipper.style.width=pos+"%";imgB.style.zIndex=3;}
    }

    function layout(){
        const cw = vContainer.clientWidth; const ch = vContainer.clientHeight;
        const aw = imgA.naturalWidth||1024; const ah = imgA.naturalHeight||768;
        const ar = aw/ah; let tw, th;
        if (cw/ch > ar) { th = ch; tw = Math.round(ch*ar); } else { tw = cw; th = Math.round(cw/ar); }
        const left = Math.round((cw - tw)/2); const top = Math.round((ch - th)/2);
        Object.assign(imgA.style,{width:tw+"px",height:th+"px",left:left+"px",top:top+"px"});
        Object.assign(imgB.style,{width:tw+"px",height:th+"px",left:left+"px",top:top+"px"});
    }
    imgA.addEventListener('load', layout);
    imgB.addEventListener('load', layout);
    window.addEventListener('resize', layout);

    vContainer.addEventListener('mousemove', (e) => {
        if (isDragging) {
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;
            panX += dx;
            panY += dy;
            updateTransform();
        } else {
            track(e);
        }
    });
    vContainer.addEventListener('mousedown', (e) => {
        if (e.button === 2) {
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
            e.preventDefault();
        }
    });
    vContainer.addEventListener('mouseup', () => { isDragging = false; });
    vContainer.addEventListener('contextmenu', e => e.preventDefault());
    vContainer.addEventListener('wheel', (e) => {
        e.preventDefault();
        const zoomSpeed = 0.1;
        const oldScale = scale;
        if (e.deltaY < 0) scale *= (1 + zoomSpeed);
        else scale /= (1 + zoomSpeed);
        scale = Math.max(0.1, Math.min(scale, 10));
        
        const rect = vContainer.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const rx = (mx - panX) / oldScale;
        const ry = (my - panY) / oldScale;
        panX = mx - rx * scale;
        panY = my - ry * scale;
        updateTransform();
    }, { passive: false });

    // Touch support (basic)
    vContainer.addEventListener('touchstart',track,false);
    vContainer.addEventListener('touchmove',track,false);
    content.querySelector('#ky-vc-reload').addEventListener('click',()=>{imgA.src=inpA.value;imgB.src=inpB.value;});
    const closeBtn=content.querySelector('#ky-vc-close');
    const fsBtn=content.querySelector('#ky-vc-fs');
    function close(){closeCompareDialog();}
    closeBtn.addEventListener('click',close);
    vcDialog.addEventListener('click',(e)=>{if(e.target===vcDialog) close();});
    fsBtn.addEventListener('click',()=>{
        const el = content;
        if (!document.fullscreenElement) { el.requestFullscreen?.(); el.classList.add('ky-vc-fullscreen'); }
        else { document.exitFullscreen?.(); el.classList.remove('ky-vc-fullscreen'); }
    });
    document.addEventListener('fullscreenchange', ()=>{
        const isFs = document.fullscreenElement === content;
        content.classList.toggle('ky-vc-fullscreen', isFs);
        layout();
    });
    vcDialog.tabIndex=-1;vcDialog.addEventListener('keydown',(e)=>{
        if(e.key==='Escape'){
            e.preventDefault();
            if(document.fullscreenElement){ document.exitFullscreen?.(); return; }
            close();
        } else if (e.key.toLowerCase()==='f') {
            e.preventDefault();
            if (!document.fullscreenElement) { content.requestFullscreen?.(); content.classList.add('ky-vc-fullscreen'); }
            else { document.exitFullscreen?.(); content.classList.remove('ky-vc-fullscreen'); }
        }
    });
    vcDialog.focus();
    layout();
}

function updateImageCompareDialog(a,b){
    if(!vcDialog) return openImageCompareDialog(a,b);
    const content = vcDialog.querySelector('.ky-vc-content');
    const inpA = content.querySelector('#ky-vc-a');
    const inpB = content.querySelector('#ky-vc-b');
    const imgA = content.querySelector('#ky-ic-a');
    const imgB = content.querySelector('#ky-ic-b');
    function toStr(x){return (Array.isArray(x)?x.join(''):x||"").toString();}
    const sa = toStr(a); const sb = toStr(b);
    inpA.value = sa; inpB.value = sb;
    imgA.src = sa; imgB.src = sb;
}
