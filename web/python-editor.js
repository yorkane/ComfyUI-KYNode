import { app } from "../../scripts/app.js";

// Constants
const MAX_CHAR_VARNAME = 50;
const LIST_THEMES = ["monokai"];
const varTypes = [
  "int",
  "boolean",
  "string",
  "float",
  "json",
  "list",
  "dict",
];
const typeMap = {
  int: 'int',
  boolean: 'bool',
  string: 'str',
  float: 'float',
  json: 'json',
  list: 'list',
  dict: 'dict',
}
const DEFAULT_TEMPLATES = {
  py: `import re, json, os, traceback
from time import strftime
def runCode():
    nowDataTime = strftime("%Y-%m-%d %H:%M:%S")
    return f"Hello ComfyUI with us today {nowDataTime}!"
r0 = runCode()

`,
};

function getPostition(node, ctx, w_width, y, n_height) {
  const margin = 5;

  const rect = ctx.canvas.getBoundingClientRect();
  const transform = new DOMMatrix()
    .scaleSelf(rect.width / ctx.canvas.width, rect.height / ctx.canvas.height)
    .multiplySelf(ctx.getTransform())
    .translateSelf(margin, margin + y);
  const scale = new DOMMatrix().scaleSelf(transform.a, transform.d);

  return {
    transformOrigin: "0 0",
    transform: scale,
    left: `${transform.a + transform.e + rect.left}px`,
    top: `${transform.d + transform.f + rect.top}px`,
    maxWidth: `${w_width - margin * 2}px`,
    maxHeight: `${n_height - margin * 2 - y - 15}px`,
    width: `${w_width - margin * 2}px`,
    height: "90%",
    position: "absolute",
    scrollbarColor: "var(--descrip-text) var(--bg-color)",
    scrollbarWidth: "thin",
    zIndex: app.graph._nodes.indexOf(node),
  };
}

function findWidget(node, value, attr = "name", func = "find") {
  return node?.widgets
    ? node.widgets[func]((w) =>
        Array.isArray(value) ? value.includes(w[attr]) : w[attr] === value
      )
    : null;
}

// Save data to workflow forced!
function saveValue() {
  app?.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState();
}

class PythonEditorWidget {
  static i_editor = 0;
  
  constructor(node, inputName, inputData) {
    // console.log("[PythonEditor] Constructor", { inputName, inputData });
    
    this.node = node;
    this.name = inputName;
    this.type = "pycode";
    this.options = { hideOnZoom: true };
    
    // Ensure default value is never null
    this._value = inputData[1]?.default || 
      `def my(a, b=1):
  return a * b
    
r0 = str(my(23, 9))`;
    
    this.codeElement = null;
    this.textarea = null;
    this.errorDiv = null;
    this.outputDiv = null;
    
    this.createElements();
    this.setupEventListeners();
  }
  
  get value() {
    return this._value;
  }
  
  set value(v) {
    // 避免重复设置相同的值
    if (this._value === v) {
      return;
    }
    
    // Ensure value is never null
    this._value = v || '';
    
    // 避免无限循环调用
    if (this.node && !this._settingFromNode) {
      this._settingFromNode = true;
      const widget = this.node.widgets?.find(w => w.name === this.name);
      if (widget && widget.value !== this._value) {
        widget.value = this._value;
      }
      this._settingFromNode = false;
    }
    
    if (this.textarea && this.textarea.value !== this._value) {
      this.textarea.value = this._value;
    }
  }
  
  createElements() {
    // 创建一个标准的容器元素
    this.codeElement = document.createElement('div');
    this.codeElement.className = 'python-editor-container';
    this.codeElement.style.width = '100%';
    this.codeElement.style.height = '100%';
    this.codeElement.style.position = 'relative';
    
    // 创建 textarea 作为编辑器
    this.textarea = document.createElement('textarea');
    this.textarea.value = this._value;
    this.textarea.style.width = '100%';
    this.textarea.style.height = '80%';
    this.textarea.style.fontFamily = 'cursive';
    this.textarea.style.fontSize = '1.1em';
    this.textarea.style.boxSizing = 'border-box';
    this.textarea.style.backgroundColor = '#000';
    this.textarea.style.color = '#2bb356ff';
    this.textarea.style.resize = 'none';
    this.textarea.style.wordBreak = 'break-word';
    this.textarea.wrap = 'soft';
    this.textarea.spellcheck = false;
    
    // 创建错误显示区域
    this.errorDiv = document.createElement('div');
    this.errorDiv.className = 'error-message';
    this.errorDiv.style.background = 'rgba(255, 0, 0, 0.1)';
    this.errorDiv.style.color = '#ff4444';
    this.errorDiv.style.padding = '4px';
    this.errorDiv.style.marginTop = '4px';
    this.errorDiv.style.borderRadius = '4px';
    this.errorDiv.style.fontFamily = 'monospace';
    this.errorDiv.style.fontSize = '12px';
    this.errorDiv.style.display = 'none';
    
    // 创建输出区域
    this.outputDiv = document.createElement('div');
    this.outputDiv.style.background = '#2a2a2a';
    this.outputDiv.style.color = '#ffffff';
    this.outputDiv.style.padding = '8px';
    this.outputDiv.style.marginTop = '4px';
    this.outputDiv.style.borderRadius = '4px';
    this.outputDiv.style.fontFamily = 'monospace';
    this.outputDiv.style.fontSize = '12px';
    this.outputDiv.style.height = '20%';
    this.outputDiv.style.overflowY = 'auto';
    
    // 将元素添加到容器
    this.codeElement.appendChild(this.textarea);
    this.codeElement.appendChild(this.errorDiv);
    this.codeElement.appendChild(this.outputDiv);
    
    this.codeElement.hidden = true;
    document.body.appendChild(this.codeElement);
    
    // 处理节点折叠
    const collapse = this.node.collapse;
    this.node.collapse = function () {
      collapse.apply(this, arguments);
      if (this.flags?.collapsed) {
        this.codeElement.hidden = true;
      } else {
        if (this.flags?.collapsed === false) {
          this.codeElement.hidden = false;
        }
      }
    }.bind(this.node);
  }
  
  setupEventListeners() {
    // 添加事件监听
    this.textarea.addEventListener('input', () => {
      this._value = this.textarea.value;
      if (this.node) {
        this.node.setDirtyCanvas(true);
      }
    });
    
    // 支持Tab键缩进
    this.textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const start = this.textarea.selectionStart;
        const end = this.textarea.selectionEnd;
        
        // 在光标位置插入4个空格
        this.textarea.value = this.textarea.value.substring(0, start) + '    ' + this.textarea.value.substring(end);
        
        // 将光标位置移动到插入空格后的位置
        this.textarea.selectionStart = this.textarea.selectionEnd = start + 4;
        
        // 触发 input 事件以更新 value
        this.textarea.dispatchEvent(new Event('input'));
      }
    });
    
  }
  
  setupErrorHandling(app) {
    if (!app.socket) {
      console.error("[PythonEditor] Socket not available");
      return;
    }
    
    app.socket.on("python_script_error", (data) => {
      console.log("[PythonEditor] Script error:", data);
      if (data.node_id === this.node.id) {
        this.showError(`Error on line ${data.line}: ${data.error}`);
      }
    });

    app.socket.on("python_script_output", (data) => {
      console.log("[PythonEditor] Script output:", data);
      if (data.node_id === this.node.id) {
        this.addOutput(data.output);
      }
    });

    app.socket.on("executed", (data) => {
      console.log("[PythonEditor] Script executed");
      if (data && data.node_id === this.node.id) {
        this.clearOutput();
      }
    });
  }
  
  draw(ctx, node, widget_width, y, widget_height) {
    const hidden =
      node.flags?.collapsed ||
      (!!this.options.hideOnZoom && app.canvas.ds.scale < 0.5) ||
      this.type === "converted-widget" ||
      this.type === "hidden";

    this.codeElement.hidden = hidden;

    if (hidden) {
      this.options.onHide?.(this);
      return;
    }

    Object.assign(
      this.codeElement.style,
      getPostition(node, ctx, widget_width, y, node.size[1])
    );
  }
  
  computeSize(...args) {
    return [500, 250];
  }
  
  getValue() {
    return this.textarea.value;
  }
  
  setValue(value) {
    this.textarea.value = value || '';
    this._value = this.textarea.value;
  }
  
  clearOutput() {
    this.outputDiv.innerHTML = '';
  }
  
  showError(message) {
    this.errorDiv.textContent = message;
    this.errorDiv.style.display = 'block';
    setTimeout(() => {
      this.errorDiv.style.display = 'none';
    }, 5000);
  }
  
  addOutput(text) {
    const outputLine = document.createElement('div');
    outputLine.textContent = text;
    this.outputDiv.appendChild(outputLine);
    this.outputDiv.scrollTop = this.outputDiv.scrollHeight;
  }
  
  serialize() {
    return this.value;
  }
  
  deserialize(value) {
    this.value = value || '';
  }
}

// Register extensions
app.registerExtension({
  name: "KYNode.KY_Eval_Python",
    async setup() {
        // 🔥 监听后端事件
        if (app.socket) {
            app.socket.on("python_editor_error", (data) => {
            console.log("收到错误事件:", data);

            // 获取当前所有节点，找到我们自定义的那个
            for (const id in app.graph._nodes_by_id) {
                const node = app.graph._nodes_by_id[id];
                if (node.comfyClass === "KY_Eval_Python") {
                    // 调用我们自己定义的组件方法
                    if (node.showError) {
                        node.showError(data.error);
                    }
                }
            }
        });
        } else {
             console.warn("[KY_Eval_Python] app.socket undefined during setup");
        }
    },
  getCustomWidgets(app) {
    return {
      PYCODE: (node, inputName, inputData, app) => {
        const widget = new PythonEditorWidget(node, inputName, inputData);
        // widget.setupErrorHandling(app);
        const varTypeList = node.addWidget(
          "combo",
          "select_type",
          "string",
          (v) => {
            // No theme setting needed
          },
          {
            values: varTypes,
            serialize: false,
          }
        );

        node.addWidget(
          "button",
          "Add Input variable",
          "add_input_variable",
          async () => {
            // Input name variable and check
            let nameInput = node?.inputs?.length
              ? `p${node.inputs.length - 1}`
              : "p0";

            const currentWidth = node.size[0];
            let tp = varTypeList.value
            nameInput = nameInput + '_' + typeMap[tp]
            node.addInput(nameInput, "*");
            node.setSize([currentWidth, node.size[1]]);
            let cv = widget.getValue();
            if(tp == 'json') {
              cv = cv + '\n' + nameInput + ' = json.loads('+ nameInput + ')'
            } else if(tp == 'list') {
              cv = cv + '\nfor i in '+ nameInput +':\n  print(i)'
            } else if(tp == 'dict') {
              cv = cv + '\nval = ' + nameInput + '["key"]'
            } else {
              cv = cv + '\n' + nameInput + ' = ' + typeMap[tp] + '('+ nameInput + ')'
            }
            widget.setValue(cv)
            saveValue();
          }
        );

        node.addWidget(
          "button",
          "Add Output variable",
          "add_output_variable",
          async () => {
            const currentWidth = node.size[0];
            // Output name variable
            let nameOutput = node?.outputs?.length
              ? `r${node.outputs.length}`
              : "r0";
            let tp = varTypeList.value
            nameOutput = nameOutput + '_' + typeMap[tp]
            node.addOutput(nameOutput, tp);
            node.setSize([currentWidth, node.size[1]]);
            let cv = widget.getValue();
            if(tp == 'json') {
              cv = cv + '\n' + nameOutput + ' = json.dumps('+ nameOutput + ')'
            } else if(tp == 'list') {
              cv = cv + '\n' + nameOutput + ' = []'
            } else if(tp == 'dict') {
              cv = cv + '\n' + nameOutput + ' = {}'
            } else {
              cv = cv + '\n' + nameOutput + ' = ' + typeMap[tp] + '('+ nameOutput + ')'
            }
            widget.setValue(cv)
            saveValue();
          }
        );

        node.onRemoved = function () {
          if (widget.codeElement) {
            widget.codeElement.remove();
          }
        };

        node.addCustomWidget(widget);
        return widget;
      },
    };
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // --- IDENode
    if (nodeData.name === "KY_Eval_Python") {
      // Node Created
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        const ret = onNodeCreated
          ? onNodeCreated.apply(this, arguments)
          : undefined;

        const node_title = await this.getTitle();
        const nodeName = `${nodeData.name}_${this.id}`;

        console.log(`Create ${nodeData.name}: ${nodeName}`);
        this.name = nodeName;

        // Create default inputs, when first create node
        if (this?.inputs?.length<2) {
          ["p0_str"].forEach((inputName) => {
            const currentWidth = this.size[0];
            this.addInput(inputName, "*");
            this.setSize([currentWidth, this.size[1]]);
          });
        }

        this.setSize([530, this.size[1]]);

        return ret;
      };

      const onDrawForeground = nodeType.prototype.onDrawForeground;
      nodeType.prototype.onDrawForeground = function (ctx) {
        const r = onDrawForeground?.apply?.(this, arguments);

        if (this?.outputs?.length) {
          for (let o = 0; o < this.outputs.length; o++) {
            const { name, type } = this.outputs[o];
            const colorType = LGraphCanvas.link_type_colors[type.toUpperCase()];
            const nameSize = ctx.measureText(name);
            const typeSize = ctx.measureText(
              `[${type === "*" ? "any" : type.toLowerCase()}]`
            );

            ctx.fillStyle = colorType === "" ? "#AAA" : colorType;
            ctx.font = "12px Arial, sans-serif";
            ctx.textAlign = "right";
            ctx.fillText(
              `[${type === "*" ? "any" : type.toLowerCase()}]`,
              this.size[0] - nameSize.width - typeSize.width,
              o * 20 + 19
            );
          }
        }

        if (this?.inputs?.length) {
          const not_showing = ["select_type", "pycode"];
          for (let i = 1; i < this.inputs.length; i++) {
            const { name, type } = this.inputs[i];
            if (not_showing.includes(name)) continue;
            const colorType = LGraphCanvas.link_type_colors[type.toUpperCase()];
            const nameSize = ctx.measureText(name);

            ctx.fillStyle = !colorType || colorType === "" ? "#AAA" : colorType;
            ctx.font = "12px Arial, sans-serif";
            ctx.textAlign = "left";
            ctx.fillText(
              `[${type === "*" ? "any" : type.toLowerCase()}]`,
              nameSize.width + 25,
              i * 20
            );
          }
        }
        return r;
      };

      // Node Configure
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (node) {
        onConfigure?.apply(this, arguments);
        if (node?.widgets_values?.length) {
          const widget_code_id = findWidget(
            this,
            "pycode",
            "type",
            "findIndex"
          );

          const widget = this.widgets[widget_code_id];
          if (widget && widget.setValue) {
            widget.setValue(this.widgets_values[widget_code_id]);
          }
        }
      };

      // ExtraMenuOptions
      const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
      nodeType.prototype.getExtraMenuOptions = function (_, options) {
        getExtraMenuOptions?.apply(this, arguments);

        const past_index = options.length - 1;
        const past = options[past_index];

        if (!!past) {
          // Inputs remove
          for (const input_idx in this.inputs) {
            const input = this.inputs[input_idx];

            if (["language", "select_type"].includes(input.name)) continue;

            options.splice(past_index + 1, 0, {
              content: `Remove Input ${input.name}`,
              callback: (e) => {
                const currentWidth = this.size[0];
                if (input.link) {
                  app.graph.removeLink(input.link);
                }
                this.removeInput(input_idx);
                this.setSize([80, this.size[1]]);
                saveValue();
              },
            });
          }

          // Output remove
          for (const output_idx in this.outputs) {
            const output = this.outputs[output_idx];

            if (output.name === "r0") continue;

            options.splice(past_index + 1, 0, {
              content: `Remove Output ${output.name}`,
              callback: (e) => {
                const currentWidth = this.size[0];
                if (output.link) {
                  app.graph.removeLink(output.link);
                }
                this.removeOutput(output_idx);
                this.setSize([currentWidth, this.size[1]]);
                saveValue();
              },
            });
          }
        }
      };
      // end - ExtraMenuOptions
    }
  },
});