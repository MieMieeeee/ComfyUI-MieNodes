# ComfyUI-MieNodes  

[English](README.md) | [简体中文](README_CN.md)  

**ComfyUI-MieNodes** 是一款为 ComfyUI 生态系统设计的插件，提供了一系列实用节点，旨在简化工作流程并提升效率。  

---

## 当前功能  

### LoRA 训练标注准备功能  

该插件提供以下实用节点，专注于 LoRA 训练流程中的数据集文件管理任务：  

1. 批量编辑标注文件（插入/追加/替换操作）。  
2. 批量重命名文件，添加前缀并格式化文件编号。  
3. 同步图像文件和标注文件，支持自动创建或删除与图像对应的 `.txt` 文件。  
4. 批量读取标注文件，支持提取所有文件内容，用于大语言模型的分析或总结。  
5. 批量转换图像文件，支持将所有图像文件转换为指定格式（`.jpg` 或 `.png`）。  
6. 批量删除具有特定扩展名和可选前缀的文件。  
7. 删除目标目录中重复（内容相同）的图像文件。 
8. 将任意数据保存为 TOML、JSON 或 TXT 格式的文件。
9. 比较两个文件（TOML或JSON格式）并返回差异。

---

### 通用功能  

插件还提供了一些常用的实用节点：  

1. 将任意输入内容以字符串形式显示。  

---

## 节点功能  

### **BatchRenameFiles**  
**功能说明：** 批量重命名文件，添加前缀和编号。  
**参数说明：**  
- `directory` (str): 目录路径。  
- `file_extension` (str): 要操作的文件扩展名，例如 `.jpg`、`.txt`。  
- `numbering_format` (str): 编号格式，例如 `###` 表示三位数字。  
- `update_caption_as_well` (bool): 是否同时重命名具有相同名称的 `.txt` 文件。  
- `prefix` (str, optional): 文件名前缀（可选）。  

---

### **BatchDeleteFiles**  
**功能说明：** 批量删除符合指定扩展名和前缀的文件。  
**参数说明：**  
- `directory` (str): 目录路径。  
- `file_extension` (str): 要删除的文件扩展名，例如 `.jpg`、`.txt`。  
- `prefix` (str, optional): 文件名前缀过滤条件（可选）。  

---

### **BatchEditTextFiles**  
**功能说明：** 对文本文件执行操作（插入、追加、替换或删除）。  
**参数说明：**  
- `directory` (str): 目录路径。  
- `operation` (str): 操作类型（`insert`、`append`、`replace`、`remove`）。  
- `file_extension` (str, optional): 要操作的文件扩展名，例如 `.txt`。  
- `target_text` (str, optional): 替换或删除的目标文本（仅用于替换或删除操作）。  
- `new_text` (str, optional): 要插入、追加或替换的新内容。  

---

### **BatchSyncImageCaptionFiles**  
**功能说明：** 为目录中的图像文件添加标注文件（同名 `.txt` 文件）。  
**参数说明：**  
- `directory` (str): 目录路径。  
- `caption_content` (str): 写入标注文件的内容，例如 `"nazha,"`。  

---

### **SummaryTextFiles**  
**功能说明：** 摘要生成当前目录下所有文本文件的数据内容。  
**参数说明：**  
- `directory` (str): 目录路径。  
- `add_separator` (bool): 是否在文件内容之间添加分隔符。  
- `save_to_file` (bool): 是否将摘要保存到文件中。  
- `file_extension` (str, optional): 要操作的文件扩展名，例如 `.txt`。  
- `summary_file_name` (str, optional): 保存摘要的文件名（可选）。  

---

### **BatchConvertImageFiles**  
**功能说明：** 将指定目录中的所有图像文件转换为目标格式。  
**参数说明：**  
- `directory` (str): 目录路径。  
- `target_format` (str): 目标图像格式（`jpg` 或 `png`）。  
- `save_original` (bool): 是否保留原始文件。  

---

### **DedupImageFiles**  
**功能说明：** 删除指定目录中的重复图像文件。  
**参数说明：**  
- `directory` (str): 目录路径。  
- `max_distance_threshold` (int): 最大 Hamming 距离阈值，用于判断图像是否重复。  

---

### **ShowAnythingMie**  
**功能说明：** 将输入内容以字符串形式打印输出。  
**参数说明：**  
- `anything` (*): 输入的任意内容。  

---

### **SaveAnythingAsFile**
**功能说明：** 将数据保存为 TOML、JSON 或 TXT 格式的文件。  
**参数说明：**  
- `data` (\*): 要保存的数据。  
- `directory` (str): 保存文件的目录。  
- `file_name` (str): 输出文件的名称。  
- `save_format` (str): 保存数据的格式（"json"、"toml" 或 "txt"）。

---

### **CompareFiles**
**功能说明：** 比较两个文件并返回差异。
**参数说明：**
- `file1_path` (str): 第一个文件的路径。
- `file2_path` (str): 第二个文件的路径。
- `file_format` (str): 文件的格式（"json" 或 "toml"）。

---

## 未来计划  

ComfyUI-MieNodes 插件正在积极开发中，未来将进一步扩展功能。规划中的新功能包括：  

- 自动生成图像的逆向标注（Reverse Caption）。  
- 支持复杂的节点链路配置，以提升数据流集成能力。  

由于作者同时是一名内容创作者，未来还将根据视频创作需求，陆续添加众多实用工具。敬请期待更多更新！  

---

## 联系方式  

- **B站**: [@黎黎原上咩](https://space.bilibili.com/449342345)  
- **YouTube**: [@SweetValberry](https://www.youtube.com/@SweetValberry)  
