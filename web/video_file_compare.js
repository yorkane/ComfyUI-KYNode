// 用于实现视频对比的JavaScript代码
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// 注册节点扩展
app.registerExtension({
    name: "KYNode.VideoCompare",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 检查是否为我们的视频对比节点
        if (nodeData.name === "KY_VideoCompare") {
            console.log("Registering KY_VideoCompare node extension");
            
            // 当节点创建时，添加一个处理函数
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                console.log("KY_VideoCompare node created");
                
                // 创建一个用于显示视频对比的预览窗口
                this.previewWidget = this.addWidget("button", "Video Compare URL", "预览", () => {
                    console.log("Preview button clicked");
                    // 获取节点的输入值
                    const mode = this.widgets.find(w => w.name === "mode")?.value || "url";
                    let videoAUrl = "";
                    let videoBUrl = "";
                    
                    if (mode === "url") {
                        videoAUrl = this.widgets.find(w => w.name === "video_a_url")?.value || "";
                        videoBUrl = this.widgets.find(w => w.name === "video_b_url")?.value || "";
                    }
                    
                    // 打开新的窗口显示视频对比
                    openVideoCompareWindow(videoAUrl, videoBUrl);
                });
                
                return r;
            };
            
            // 处理来自节点的UI输出
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                console.log("KY_VideoCompare onExecuted called with message:", message);
                onExecuted?.apply(this, arguments);
                
                // 检查消息中是否包含我们需要的数据
                if (message && typeof message === 'object') {
                    // 确保我们获取的是字符串而不是数组
                    let videoAUrl = message.video_a_source || "";
                    let videoBUrl = message.video_b_source || "";
                    const mode = message.mode || "url";
                    
                    // 如果是数组，转换为字符串
                    if (Array.isArray(videoAUrl)) {
                        videoAUrl = videoAUrl.join('');
                        console.log("Converted videoAUrl from array to string:", videoAUrl);
                    }
                    if (Array.isArray(videoBUrl)) {
                        videoBUrl = videoBUrl.join('');
                        console.log("Converted videoBUrl from array to string:", videoBUrl);
                    }
                    
                    console.log("Extracted data from message:");
                    console.log("Video A URL:", videoAUrl);
                    console.log("Video B URL:", videoBUrl);
                    
                    // 只有当至少有一个视频URL不为空时才打开窗口
                    if (videoAUrl || videoBUrl) {
                        console.log("Opening video compare window from onExecuted");
                        openVideoCompareWindow(videoAUrl, videoBUrl);
                    } else {
                        console.log("Both video URLs are empty, not opening window");
                    }
                } else {
                    console.log("Message is not a valid object or is empty");
                }
            };
        }
    }
});

// 打开视频对比窗口的函数
function openVideoCompareWindow(videoAUrl, videoBUrl) {    
    // 处理undefined或null值并确保是字符串
    videoAUrl = (videoAUrl || "").toString();
    videoBUrl = (videoBUrl || "").toString();
    
    // 如果是数组，转换为字符串
    if (Array.isArray(videoAUrl)) {
        videoAUrl = videoAUrl.join('');
    }
    if (Array.isArray(videoBUrl)) {
        videoBUrl = videoBUrl.join('');
    }
    
    // 创建一个新窗口显示视频对比
    const htmlContent = `
<html><head>
  <meta charset="UTF-8">
    <title>Video Compare</title>
<style>
body {
  background: #333;
  margin: 4px;
}
#video-compare-container {
  display: inline-block;
  line-height: 0;
  position: relative;
  width: 100%;
  padding-top: 42.3%;
}
#video-compare-container > video {
  width: 100%;
  position: absolute;
  top: 0; height: 100%;
}
#video-clipper {
  width: 50%; position: absolute;
  top: 0; bottom: 0;
  overflow: hidden;
  border-right: 1px dashed #ccc;
}
#video-clipper video {
  width: 200%;
  position: absolute;
  height: 100%;
}
#title1 {
    color: #ccc;
    float: right;
}
#title2 { 
    color: #ccc;
    float: left;
}
.controls {
  margin: 5px 0;
  text-align: center;
  display: flex;
  align-items: center;
  gap: 10px;
}
.controls button {
  padding: 5px;
  background: #444;
  color: white;
  border: 1px solid #666;
  cursor: pointer;
  flex-shrink: 0;
}
.controls button:hover {
  background: #555;
}
.controls input {
  padding: 4px;
  flex: 1;
  max-width: 49%;
}
</style>
  
</head>

<body translate="no">
  <div class="controls">
    <button id="reload-btn">Reload</button>
    <input type="text" id="videoA-url" placeholder="Video A URL" value="https://s3-us-west-2.amazonaws.com/s.cdpn.io/4273/floodplain-dirty.mp4">
    <input type="text" id="videoB-url" placeholder="Video B URL" value="https://s3-us-west-2.amazonaws.com/s.cdpn.io/4273/floodplain-clean.mp4">
  </div>
  
  <div id="video-compare-container">
  <video loop="" autoplay="" muted="">
    <source src="${videoAUrl}">
  </video>
 <div id="video-clipper">
    <video loop="" autoplay="" muted="" style="width: 200%; z-index: 3;">
      <source src="${videoBUrl}">
    </video>
  </div>
	</div>
      <script id="rendered-js">
function trackLocation(e) {
  var rect = videoContainer.getBoundingClientRect(),
  position = (e.pageX - rect.left) / videoContainer.offsetWidth * 100;
  if (position <= 100) {
    videoClipper.style.width = position + "%";
    clippedVideo.style.width = 100 / position * 100 + "%";
    clippedVideo.style.zIndex = 3;
  }
}

function reloadVideos() {
  var videoAUrl = document.getElementById('videoA-url').value;
  var videoBUrl = document.getElementById('videoB-url').value;
  
  var videoContainer = document.getElementById("video-compare-container");
  var videoA = videoContainer.getElementsByTagName("video")[0];
  var videoB = clippedVideo;
  
  videoA.getElementsByTagName('source')[0].src = videoAUrl;
  videoB.getElementsByTagName('source')[0].src = videoBUrl;
  
  videoA.load();
  videoB.load();
}

var videoContainer = document.getElementById("video-compare-container"),
videoClipper = document.getElementById("video-clipper"),
clippedVideo = videoClipper.getElementsByTagName("video")[0];
videoContainer.addEventListener("mousemove", trackLocation, false);
videoContainer.addEventListener("touchstart", trackLocation, false);
videoContainer.addEventListener("touchmove", trackLocation, false);
var videoA = videoContainer.getElementsByTagName("video")[0];
var videoB = clippedVideo

document.getElementById('reload-btn').addEventListener('click', reloadVideos);
        // 视频加载完成后同步播放
        videoA.addEventListener('loadeddata', function() {
            if (videoB.readyState >= 3) {
                videoA.play();
                videoB.play();
            }
        });
        videoB.addEventListener('loadeddata', function() {
            if (videoA.readyState >= 3) {
                videoA.play();
                videoB.play();
            }
        });
        // 处理视频播放同步
        function syncVideos() {
            if (videoA.paused && videoB.paused) return;
            
            const diff = Math.abs(videoA.currentTime - videoB.currentTime);
            if (diff > 0.1) {
                // 同步视频时间
                videoB.currentTime = videoA.currentTime;
            }
        }
        // 定期同步视频
        setInterval(syncVideos, 100);
</script></body></html>
    `;
    
    // 创建新窗口
    const newWindow = window.open("", "_blank");
    newWindow.document.write(htmlContent);
    newWindow.document.close();
}