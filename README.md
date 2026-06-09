# ComfyUI-MieNodes  

[English](README.md) | [简体中文](README_CN.md)  

**ComfyUI-MieNodes** is a plugin for the ComfyUI ecosystem, offering a series of utility nodes designed to simplify workflows and enhance efficiency.  

---

## Workflows

Currently, the following services are supported:
  - [ZhiPu 智谱](https://www.bigmodel.cn/glm-coding?ic=QCHZLYWEXV) — standard + Coding/Token Plan tiers
  - [SiliconFlow 硅基流动](https://cloud.siliconflow.cn/i/PYyJkS9S)
  - [GitHub Models](https://github.com/marketplace?type=models)
  - [Kimi](https://platform.moonshot.cn)
  - [DeepSeek](https://platform.deepseek.com)
  - [Gemini](https://ai.google.dev/gemini) — multimodal-aware (image input is forwarded as inline data)
  - [Bailian 阿里云百炼](https://bailian.console.aliyun.com/)
  - [MiniMax](https://api.minimaxi.com/) — standard Open Platform + Token Plan (`sk-cp-...` keys)
  - Any OpenAI-compatible endpoint via `SetGeneralLLMServiceConnector` (custom base URL + model)

If you wish to use other large language model (LLM) services that cannot be connected through SetGeneralLLMServiceConnector, please submit an issue or a pull request for feedback.
 
Among them, Zhipu AI's GLM-4-Flash-250414 and SiliconFlow's Qwen3-8B, GLM-Z1-9B-0414, and GLM-4-9B-0414 are free models. You only need to register and obtain an API Key (or Token) to use them without restriction.

### Call LLM Service Workflow
![Image](images/CallLLMService.png)

This workflow demonstrates how to connect to an LLM service and generate a response based on a given prompt, you can use any LLM service supported.

### Kontext Presets Prompt Generator Workflow

![Image](images/KontextPrompt.png)

This workflow demonstrates how to load an image, generate a detailed description using the Florence2 model, and compose context prompts with the help of large language models. The results are displayed both in English and Chinese using the `Show Anything` node.  
- **Image Loader**: Loads the input image.
- **Florence2 Model Loader & Describe Image**: Generates a detailed caption for the image.
- **Set SiliconFlow LLM Service Connector & Kontext Prompt Generator**: Uses LLM to create context-aware prompts.
- **Show Anything**: Displays the generated result.

### Check LLM Service Connection Workflow
![Image](images/CheckLLMServiceConnection.png)

This workflow checks the connection to the LLM service and displays the result.

### Kontext Presets Add and Remove Workflow

![Image](images/AddAndRemoveUserPresets.png)

This workflow demonstrates how to add and remove a custom preset.

### Advanced Prompt Generator Workflow

![Image](images/PromptGenerator.png)

This workflow focuses on generating an artistic prompt for an ethereal digital portrait using the LLM connector and advanced prompt generator nodes.  
- **Prompt Generator**: Produces detailed prompts for creative tasks.
- **Show Anything**: Displays the generated result.

---

### Bernini Prompt Generator Workflow

![Image](images/BerniniPromptGenerator.png)

This workflow routes a user prompt through the [Bernini](https://bernini-ai.github.io/) task-aware prompt enhancer (bytedance/Bernini, Apache 2.0) and writes an enriched, task-shaped prompt back into the graph. The 13 task types supported in the dropdown are:

- **t2i / t2v** — text-to-image / text-to-video
- **i2i / i2v** — image-to-image editing / image-to-video
- **r2i / r2v** — subject-driven image / video generation (reference image of a subject)
- **ri2i** — reference-image-guided image editing (source image + reference image; MieNodes extension)
- **v2v / mv2v** — video-to-video editing / multi-source v2v
- **vi2v** — video editing with a reference image
- **rv2v / vrc2v** — reference-guided video editing
- **ads2v** — ads insertion in video

Reference images and source video frames are forwarded to the LLM as `image_url` content parts, so the model can see what it's rewriting about. The dropdown labels are bilingual (code + 中文) and stay readable in either language.

---
## Current Features  

### Prompt Enhancement Features

The plugin provides a suite of nodes for prompt enhancement, offering:

1. A collection of preset workflows that leverage large language models to automatically generate high-quality prompts from image and text inputs.
2. Advanced prompt optimization, including automatic translation and enrichment of details, enabling richer, more expressive outputs for various creative tasks.

---

### LoRA Training Caption Preparation Features  

The plugin provides the following utility nodes, with a focus on dataset file management tasks in LoRA training workflows:  

1. Batch edit caption files (Insert/Append/Replace operations).  
2. Batch rename files, add prefixes, and format file numbering for specific file types.  
3. Synchronize image and caption files, with support for automatically creating or removing `.txt` files to match image files.  
4. Batch read caption files, with support for extracting all file contents for analysis and summarization by large language models.  
5. Batch convert image files, enabling conversion of all image files to a specified format (`.jpg` or `.png`).  
6. Batch delete files with the specified extension and optional prefix.  
7. Remove duplicated (same content) image files in the specified directory. 
8. Save any data as a file in TOML, JSON, or TXT format.
9. Compare two files (in TOML or JSON format) and return the differences.

---

### Common Features  

The plugin also provides utility nodes for general-purpose tasks:  

1. Display any input as a string.
2. Download files from huggingface, hf-mirror, github or anywhere to models folder. 
3. Connect to large language models (LLMs) for advanced prompt generation, enhancement, translation, and style adaptation.
4. Translate any text to another language (English, Chinese, etc.).

---

## Nodes  

### **BatchRenameFiles**  
**Function:** Batch rename files and add a prefix and numbering.  
**Parameters:**  
- `directory` (str): Path to the directory.  
- `file_extension` (str): File extension to operate on (e.g., `.jpg`, `.txt`).  
- `numbering_format` (str): Numbering format (`###` means three digits).  
- `update_caption_as_well` (bool): Whether to also rename `.txt` files with the same name.  
- `prefix` (str, optional): Prefix to add to the file name.  

![Image](images/BatchRenameFiles.png)

---

### **BatchDeleteFiles**  
**Function:** Batch delete files with the specified extension and optional prefix.  
**Parameters:**  
- `directory` (str): Path to the directory.  
- `file_extension` (str): File extension to delete (e.g., `.jpg`, `.txt`).  
- `prefix` (str, optional): Prefix to check before deleting files.  

![Image](images/BatchDeleteFiles.png)

Before:
![Image](images/BatchDeleteFiles-1.png)

After:
![Image](images/BatchDeleteFiles-2.png)

---

### **BatchEditTextFiles**  
**Function:** Perform operations on text files (Insert, Append, Replace, or Remove).  
**Parameters:**  
- `directory` (str): Path to the directory.  
- `operation` (str): Type of operation (`insert`, `append`, `replace`, `remove`).  
- `file_extension` (str, optional): File extension to operate on (e.g., `.txt`).  
- `target_text` (str, optional): Text to replace or remove (only used for Replace or Remove operations).  
- `new_text` (str, optional): New content to insert, append, or replace.  

![Image](images/BatchEditTextFiles.png)

Before:
![Image](images/BatchEditTextFiles-1.png)

After:
![Image](images/BatchEditTextFiles-2.png)

---

### **BatchSyncImageCaptionFiles**  
**Function:** Add caption files (`.txt` files with the same name) for image files in a directory.  
**Parameters:**  
- `directory` (str): Path to the directory.  
- `caption_content` (str): Content to populate in the caption file (e.g., `"nazha,"`).  

![Image](images/BatchSyncImageCaptionFiles.png)

Before:
![Image](images/BatchSyncImageCaptionFiles-1.png)

After:
![Image](images/BatchSyncImageCaptionFiles-2.png)

---

### **SummaryTextFiles**  
**Function:** Summarize the content of all text files in a directory.  
**Parameters:**  
- `directory` (str): Path to the directory.  
- `add_separator` (bool): Whether to add a separator between file contents.  
- `save_to_file` (bool): Whether to save the summarized content to a file.  
- `file_extension` (str, optional): File extension to operate on (e.g., `.txt`).  
- `summary_file_name` (str, optional): Name of the file to save the summary.  

![Image](images/SummaryTextFiles.png)

---

### **BatchConvertImageFiles**  
**Function:** Convert all images in a specified directory to the target format.  
**Parameters:**  
- `directory` (str): Path to the directory.  
- `target_format` (str): Target image format (`jpg` or `png`).  
- `save_original` (bool): Whether to retain the original files after conversion.  

![Image](images/BatchConvertImageFiles.png)

Before:
![Image](images/BatchConvertImageFiles-1.png)

After:
![Image](images/BatchConvertImageFiles-2.png)

---

### **DedupImageFiles**  
**Function:** Remove duplicate image files in a specific directory.  
**Parameters:**  
- `directory` (str): Path to the directory.  
- `max_distance_threshold` (int): Maximum Hamming distance threshold for identifying duplicates.  

![Image](images/DedupImageFiles.png)

Before:
![Image](images/DedupImageFiles-1.png)

After:
![Image](images/DedupImageFiles-2.png)

---

### **ShowAnythingMie**  
**Function:** Print the input content as a string.  
**Parameters:**  
- `anything` (*): The input content to display.  

---

### **SaveAnythingAsFile**
**Function:** Save data to a file in either TOML, JSON, or TXT format.  
**Parameters:**  
- `data` (\*): The data to save.  
- `directory` (str): The directory to save the file in.  
- `file_name` (str): The name of the output file.  
- `save_format` (str): The format to save the data in ("json", "toml", or "txt").

---

### **CompareFiles**
**Function:** Compare two files and return the differences.
**Parameter:**
- `file1_path` (str): The path to the first file.
- `file2_path` (str): The path to the second file.
- `file_format` (str): The format of the files ("json" or "toml").

![Image](images/CompareFiles.png)

---

### **ModelDownloader**
**Function:** Download files from Hugging Face, hf-mirror, GitHub, or any other source to the models folder.
**Parameters:**
- `url` (str): The URL of the file to download.
- `save_path` (str): The path to save the downloaded file.
- `override` (bool): Whether to override the existing file if it already exists.
- `use_hf_mirror` (bool): Whether to use the Hugging Face mirror URL.
- `rename_to` (str, optional): The new name for the downloaded file (optional).
- `hf_token` (str, optional): The Hugging Face token for authentication (optional).
- `trigger_signal` (\*, optional): A signal to trigger the download (optional).

![Image](images/downloader.png)

### **SetMiniMaxLLMServiceConnector** (standard, not token plan)
**Function:** Connect to the MiniMax Open Platform API with a standard `eyJ...` (JWT) key.
**Parameters:**
- `api_token` (str): API key, or leave empty to use the `minimax_open_platform` entry in `mie_llm_keys.json`.
- `model_select`: one of `MiniMax-M2.7`, `MiniMax-M2.7-highspeed`, `MiniMax-M2.5`, `MiniMax-M2.5-highspeed`, `Custom`. Default: `MiniMax-M2.7`.
- `config_file` / `config_key` / `prefer_local_config`: standard config plumbing.

### **SetMiniMaxTokenPlanLLMServiceConnector** (Token Plan)
**Function:** Connect to the MiniMax Token Plan / Coding Plan endpoint with an `sk-cp-...` prefixed key.
**Parameters:**
- `api_token` (str): Token Plan API key, or leave empty to use the `minimax` entry in `mie_llm_keys.json`.
- `model_select`: `MiniMax-M3` (default), `MiniMax-M2.7`, `MiniMax-M2.7-highspeed`, `MiniMax-M2.5`, `MiniMax-M2.5-highspeed`, `Custom`.
- `config_file` / `config_key` / `prefer_local_config`: standard config plumbing.

The MiniMax connectors share a per-connector image-detail sanitization step that drops `detail: "auto"` from `image_url` parts before sending — MiniMax rejects that value with HTTP 400. The OpenAI-compat services (SiliconFlow, ZhiPu, Kimi, etc.) are unaffected by this.

### **BerniniPromptGenerator**
**Function:** Rewrite a user prompt using a Bernini task-aware system prompt. Optionally forward media (1 source image or a batch of source video frames, plus 0+ reference images) so the LLM can see what it's rewriting about.
**Parameters:**
- `llm_service_connector` (`LLMServiceConnector`): any connector from the list above.
- `task_type`: one of the 13 task types (dropdown shows `code - 中文` labels). Default: `t2i - 文生图`. The `ri2i (扩展)` entry is a MieNodes extension task.
- `user_prompt` (str): the raw instruction to rewrite.
- `source_images` (`IMAGE`, optional): the thing being operated on. For image-source tasks (`i2i` / `ri2i` / `i2v`) pass a single image; for video-source tasks (`v2v` / `mv2v` / `vi2v` / `rv2v` / `vrc2v` / `ads2v`) pass a batch of video frames. Unused by `r2i` / `r2v`.
- `reference_images` (`IMAGE`, optional): 0+ reference images. The subject for `r2i` / `r2v`, the guidance material for `ri2i` / `vi2v` / `rv2v` / `vrc2v`, extra context for `i2i` / `i2v`.
- `video_frames` (int, 1-16): how many of `source_images` to forward when the task treats it as video frames (ignored for image-source tasks). Default: 3.
- `image_detail` (`auto` / `low` / `high`): OpenAI-style image detail hint.
- `temperature` / `top_p` / `max_tokens`: standard sampling parameters.

Routing detail:
- `t2i` and `t2v` use specialized Chinese system prompts (`T2I_A14B_EN_SYS_PROMPT`, `T2V_A14B_EN_SYS_PROMPT`).
- The other 11 tasks use the per-task system prompt in `bernini_prompts.SYSTEM_PROMPTS` and route reference material through the appropriate `*_TEMPLATE`.
- `r2i`, `r2v`, `rv2v`, `vrc2v`, `ri2i` return JSON-mode outputs (`{"rewritten_text": "..."}`); the node parses the JSON and returns the inner string.

---
## Future Plans  

ComfyUI-MieNodes is under active development and will expand its features in future updates. Planned additions include:  

- Automatic reverse caption generation for images.  
- Support for complex node chaining to improve data flow integration.  

As the author is a content creator, the plugin will also include many practical tools developed to address specific video production needs. Stay tuned for more updates!  

---

## Contact Me  

- **Bilibili**: [@黎黎原上咩](https://space.bilibili.com/449342345)  
- **YouTube**: [@SweetValberry](https://www.youtube.com/@SweetValberry)  


---

## Acknowledgments

Some features in this project are inspired by, or build directly on the work of, the following open-source projects. We are grateful to the original authors and contributors.

- **[Bernini](https://bernini-ai.github.io/)** by the Bernini Team at ByteDance (Chenchen Liu, Junyi Chen, Lei Li, Lu Chi, Mingzhen Sun, Zhuoying Li, Yi Fu, Ruoyu Guo, Yiheng Wu, Ge Bai, Zehuan Yuan). The 12 task-aware system prompts and user templates used by the `BerniniPromptGenerator` node (under [`nodes/llm/bernini_prompts.py`](nodes/llm/bernini_prompts.py)) are a verbatim copy of the upstream [bytedance/Bernini](https://github.com/bytedance/Bernini) prompt library (Apache 2.0). Routing logic in [`bernini_prompt_generator.py`](nodes/llm/bernini_prompt_generator.py) is a ComfyUI-friendly port of `bernini.prompt_enhancer.PromptEnhancer` from the same upstream repo.

- **[rgthree-comfy](https://github.com/rgthree/rgthree-comfy)** by [@rgthree](https://github.com/rgthree).
  The SimpleText and RichText canvas annotation nodes (in [js/textNodes.js](js/textNodes.js) and [
odes/common/text_nodes.py](nodes/common/text_nodes.py)) are a direct re-implementation of rgthree's Label node. The transparent LiteGraph shell, the LGraphCanvas.prototype.drawNode wrapper, the Canvas 
oundRect/draw(ctx) rendering, and the no-op INPUT_TYPES / 
oop Python shell all follow rgthree's approach. RichText reuses the same idea and adds a DOM widget for sanitized Markdown rendering. Many thanks to rgthree for the clean reference implementation and for the years of work that have made ComfyUI's node ecosystem so much better.
