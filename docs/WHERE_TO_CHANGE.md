# 改功能或修 bug 时，改哪个仓库

短剧这条线已经拆开。一个症状只改一个地方。改错仓库时，节点壳和网页会继续调用旧逻辑。

同一份说明在引擎：`F:\ComfyUI-Mie-H3-Engine\docs\WHERE_TO_CHANGE.md`。

## 五个落点

| 你在改什么 | 仓库 | 代码在哪 |
|---|---|---|
| 平台、密钥、请求、重试、预算、多模态 data URL | `F:\ComfyUI-Mie-LLM` | `mie_llm/`（`client.py`、`platforms.py`、`budget.py`、`media.py`） |
| 提示词正文、族模板、loader、翻译用的语言表、宽高比字符串 | `F:\ComfyUI-Mie-Prompt` | `mie_prompt/` 与 `mie_prompt/prompts/<族>/` |
| 资产怎么落盘、剧本契约、零 LLM 组装、critic、对白、loop 增强的文本、镜头列表规则 | `F:\ComfyUI-Mie-Craft` | `mie_craft/`（`assets/`、`script/`、`h3_loop_prompts.py`、`loop_enhancer.py`） |
| H3 链怎么跑、checkpoint、缓存、恢复、`\|MieH3` 节点、预览路由、场景编辑器、短剧工作台页面 | `F:\ComfyUI-Mie-H3-Engine` | 仓库根的链代码、`web/`、`webapp/`、`mie_nodes.py`、`plan_preview.py` |
| ComfyUI 上的节点壳：插座、控件、tensor 和 data URL 互转、`\|Mie` 显示名、本包自己的前端和仍挂在 ComfyUI 上的 HTTP | 本仓库 | `nodes/`、`js/`、`services/llm.py`（壳） |

依赖方向是 LLM ← Prompt ← Craft。引擎和本仓库都调用这三个包。三个包不得 import ComfyUI。

## 怎么判断

- 提示词的句子、白名单、某个族的模板错了：改 `mie_prompt`。本仓库 `nodes/llm/*_prompts.py` 是 shim，改它不会改变生成结果。
- 资产 json、图片文件名、剧本校验、`assemble_plan` 的计划正文：改 `mie_craft`。工作台 `webapp/services/library.py` 只是在调用它。
- 连不上某个平台、密钥、finish_reason、预算重试：改 `mie_llm`。本仓库 `services/llm.py` 是 `from mie_llm.client import *`。
- 链上的缓存、mtime、checkpoint、恢复、导出校验：改引擎，不要改上游 `ComfyUI-MiniMaxH3-Context-Loop`。
- 图还没跑就能看到剧本计划、`plan_json_input` 接上后编辑器只读：改引擎的 `plan_preview.py` 和 `web/mieh3_plan_preview.js`，以及两个 scene editor。
- 短剧网页的页面、路由、任务记录、`/api/health` 的 `idle`：改引擎的 `webapp/`。启动方式是在引擎根目录执行 ComfyUI 自带的 `python webapp\__main__.py`，端口 8199。
- 节点多了一个插座、控件文案、前端下拉：改本仓库的对应 `nodes/**` 和 `js/`。逻辑仍放在包里，壳只负责把 ComfyUI 的类型接进去。
- `POST /mie_nodes/assets/delete` 固定返回 405，指向工作台。队列里的删除节点仍调用 `mie_craft`。不要把删除做回这个 HTTP 路由。

`shot_list_errors`（镜头列表必须是 1..128）在 `mie_craft.h3_loop_prompts`。引擎的 `_normalize_plan` 调用它。分节 prompt 规则只留在 `validate_plan`，因为链计划的镜头 prompt 是普通字符串，套分节规则会把现有链计划全部判失败。

## 本机路径，不要改错树

- 本仓库的开发副本是 `C:\Users\administered\PycharmProjects\ComfyUI-MieNodes`。
- ComfyUI 正在加载的 `custom_nodes\comfyui_mienodes` **不是**这个开发仓的 junction。改这里不会自动进正在跑的 ComfyUI。要测本仓库的 HTTP 或节点壳，得把开发仓装进那个 custom_nodes，或对本仓库跑 pytest。
- 引擎的 ComfyUI 副本是 junction：`custom_nodes\ComfyUI-Mie-H3-Engine` → `F:\ComfyUI-Mie-H3-Engine`。改 F: 盘，重启 ComfyUI 即生效。
- 系统 Python 和 `C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\python_embeded\python.exe` 都用可编辑安装指向 `F:\ComfyUI-Mie-LLM`、`F:\ComfyUI-Mie-Prompt`、`F:\ComfyUI-Mie-Craft`。改这三个仓库的源码，两个解释器同时看到。发 PyPI 是另一次发布，不是每次修改的必做步骤。
- ComfyUI 的启停走 `C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\ComfyUI启动器-CLI.cmd` 的 `status --json` / `start` / `stop`。

未跟踪的 `nodes/web/short_drama_webui.py` 会执行 `python -m mie_web`。没有这个包。工作台不从这里启动。
