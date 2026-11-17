# 待测试：拆分之后的短剧工作台与 ComfyUI

给本地 agent。按顺序做。每条记下实际 HTTP 状态、页面上看到的句子、以及失败时的响应正文。不要把「进程在」写成「功能通过」。

已经在 2026-11-17 这台机器上做过、不必当作未测的：

- ComfyUI `http://127.0.0.1:8188` 曾成功执行 `MieH3LoadScriptAsset`（剧本「Claude的千层套路001-系列版」）。
- `MiniMaxH3ChainExportPNG|MieH3` 的 `checkpoint_verification` 默认是 `strict`。
- `POST /mieh3/plan_preview` 拒绝 `../secret.json` 和 `C:/Windows/win.ini`（都是 400）。同一接口对「Claude的千层套路001-系列版」返回 422：`character_1 (Character 1) requires a character asset`。
- `tests/test_asset_library.py` 74 项通过。那是 pytest，不是网页。

工作台页面没有点过。下面第 2 节是这次的主体。

## 0. 开始前

- ComfyUI 根目录：`C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large`
- 启停：`ComfyUI启动器-CLI.cmd status --json` / `start` / `stop`。`start` 可能在进程还没听端口时就返回；以 `http://127.0.0.1:8188/system_stats` 为准。
- 引擎源码：`F:\ComfyUI-Mie-H3-Engine`（`custom_nodes\ComfyUI-Mie-H3-Engine` 是指向它的 junction）。
- MieNodes 开发仓：`C:\Users\administered\PycharmProjects\ComfyUI-MieNodes`。正在被 ComfyUI 加载的 `custom_nodes\comfyui_mienodes` 是另一份目录，不是这个开发仓。
- 工作台从引擎根目录启动：

```
C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\python_embeded\python.exe webapp\__main__.py
```

工作目录必须是 `F:\ComfyUI-Mie-H3-Engine`。端口 8199。若 8199 已有旧进程，先停掉再拉这一份。Package Launcher 的工作台也想占 8199，不要两个一起开。

- 资产库：`C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\ComfyUI\custom_nodes\comfyui_mienodes\asset_library`
- 改完引擎或工作台的 Python 后要重启对应进程。三个 pip 包是可编辑安装，改 `F:\ComfyUI-Mie-*` 源码不用重新 pip install，但已经启动的进程要重启才读到新代码。

## 1. 进程是否是这次的代码

1. `GET http://127.0.0.1:8199/api/health`
   - 通过：`status` 为 `ok`，`app` 为 `comfyui-mie-web`，并且有 `idle` 字段。没有 `idle` 说明跑的是旧进程。
   - 没有渲染任务时 `idle` 为 `true`。
2. `GET http://127.0.0.1:8188/object_info/MieH3ScriptToPlan` 返回 200。
3. `POST http://127.0.0.1:8188/mieh3/plan_preview`，正文 `{"script_ref":"../secret.json"}`，得到 400，正文含 `..`。

## 2. 工作台页面（必测）

浏览器打开 `http://127.0.0.1:8199`。逐页做，不要只看 HTML 是否返回 200。

### 2.1 `/llm`

- 页面列出平台，当前平台和密钥文件路径可见。
- 选一个已经配过密钥的平台做连通性测试。记下成功或服务端返回的原句。不要为了测试去新建密钥。

### 2.2 `/assets`

- 列表能看到已有角色和场景（库里大约 58 个角色、28 个场景）。
- 新建一个名字只用于测试的角色，例如 `验收探针`。上传一张 PNG，填 description 和 voice，保存。
- 通过：磁盘上 `asset_library\characters\验收探针\character.json` 含 `media.images`，里面的文件名是 64 位十六进制加 `.png`。`images\` 里先有这个文件，清单指向它。没有 `01.png` 也算通过。
- 只改 description 再保存一次。通过：图片哈希文件名不变，description 已更新，旧图还在。
- 打开这个角色再保存时，如果页面能带上一次返回的 `content_hash`，故意用一个错的 `expected_hash` 再存。通过：HTTP 409，正文有 `code: conflict`，磁盘内容没被盖掉。页面如果还不会发送 `expected_hash`，把这一条记成「接口已有，页面未接」，不要写成失败。
- 测完删除这个探针角色，用工作台自己的删除，不要留在库里。

### 2.3 `/scripts`

- 打开已有剧本「Claude的千层套路001-系列版」。页面能看到角色和分镜，而不是空白或「损坏」。
- 不要改这个剧本的正文。若页面有校验，确认它给出的是契约提示，而不是把文件写坏。
- 另存或新建一个探针剧本再删掉。不要覆盖库里已有的 26 个剧本。

### 2.4 `/video`

- 选「Claude的千层套路001-系列版」。
- 角色没有接上资产时，页面应拒绝或明确报缺资产，而不是提交一个空计划。这和预览接口的 422 是同一条规则。
- 给出场角色接上库里已有的对应资产（Claude娘一类已有立绘的角色）。生成计划预览。通过：预览里能看到镜头，而不是工作流 json 里那个占位句 “Describe the opening shot.”。
- 提交一次渲染。通过的最低线：任务出现在列表里；失败时列表里是 `error` 和原因，而不是永远 `running`，也不是点了没反应。
- 不要求这次渲染出成片。若 ComfyUI 排队后真实出片，记下 run name、耗时和产物路径。若失败，保存响应和 `ComfyUI\user\comfyui.log` 里对应的报错。

### 2.5 `/history`

- 刚提交的那条任务在重启工作台之后还在。做法：记下 job id，停掉 8199 的 python，再用第 0 节的命令拉起，刷新历史页。
- 通过：该条还在，状态不是被重启洗掉。

## 3. ComfyUI 上补两下

1. 用开发仓里的说明核对删除路由。这一条不要打到 `custom_nodes\comfyui_mienodes`，除非确认那份目录已经是开发仓。对开发仓的行为，pytest 已覆盖 `test_delete_endpoint_happy_missing_and_sentinel`：`POST /mie_nodes/assets/delete` 为 405，角色目录还在。
2. 在 ComfyUI 图里把 `MieH3LoadScriptAsset` 接到 `MieH3ScriptToPlan`，再把 `plan_json` 接到带 `|MieH3` 的 Plan 节点的 `plan_json_input`。打开 Scene Prompt Editor。通过：横幅写明外部 plan 接管、只读，并有 `当前预览 =` 后 8 位十六进制，或者在缺资产时显示 422 的那句缺资产，而不是静默显示 widget 里的旧 JSON。编辑器里打字不能把 widget 改掉。

## 4. 明确不要在这次做的

- 不要跑完整的 `MIE_LIVE_E2E=1` 组合矩阵，除非这次页面测试都过了还要加测提示词。那套测试会逐条调 MiniMax-M3，每条最长 900 秒。
- 不要为了浏览器 `.mjs` 去装 `/usr/bin/google-chrome-stable`。那是 Linux 路径。本机 Chrome 在 `C:\Program Files\Google\Chrome\Application\chrome.exe`。
- 不要把 `nodes/web/short_drama_webui.py` 当成工作台入口。

## 5. 结果怎么写

交一份短记录，每节一行：通过 / 失败 / 页面未接。失败写 URL、状态码、页面原句或日志原文。探针角色和探针剧本测完即删。
