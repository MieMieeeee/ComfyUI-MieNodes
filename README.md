# ComfyUI-MieNodes  

[English](README.md) | [简体中文](README_CN.md)  

**ComfyUI-MieNodes** is a plugin for the ComfyUI ecosystem, offering a series of utility nodes designed to simplify workflows and enhance efficiency.  

---

## Current Features  

### LoRA Training Caption Preparation Features  

The plugin provides the following utility nodes, with a focus on dataset file management tasks in LoRA training workflows:  

1. Batch edit caption files (Insert/Append/Replace operations).  
2. Batch rename files, add prefixes, and format file numbering for specific file types.  
3. Synchronize image and caption files, with support for automatically creating or removing `.txt` files to match image files.  
4. Batch read caption files, with support for extracting all file contents for analysis and summarization by large language models.  
5. Batch convert image files, enabling conversion of all image files to a specified format (`.jpg` or `.png`).  
6. Batch delete files with the specified extension and optional prefix.  
7. Remove duplicated (same content) image files in the specified directory.  

---

### Common Features  

The plugin also provides utility nodes for general-purpose tasks:  

1. Display any input as a string.  

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

---

### **BatchDeleteFiles**  
**Function:** Batch delete files with the specified extension and optional prefix.  
**Parameters:**  
- `directory` (str): Path to the directory.  
- `file_extension` (str): File extension to delete (e.g., `.jpg`, `.txt`).  
- `prefix` (str, optional): Prefix to check before deleting files.  

---

### **BatchEditTextFiles**  
**Function:** Perform operations on text files (Insert, Append, Replace, or Remove).  
**Parameters:**  
- `directory` (str): Path to the directory.  
- `operation` (str): Type of operation (`insert`, `append`, `replace`, `remove`).  
- `file_extension` (str, optional): File extension to operate on (e.g., `.txt`).  
- `target_text` (str, optional): Text to replace or remove (only used for Replace or Remove operations).  
- `new_text` (str, optional): New content to insert, append, or replace.  

---

### **BatchSyncImageCaptionFiles**  
**Function:** Add caption files (`.txt` files with the same name) for image files in a directory.  
**Parameters:**  
- `directory` (str): Path to the directory.  
- `caption_content` (str): Content to populate in the caption file (e.g., `"nazha,"`).  

---

### **SummaryTextFiles**  
**Function:** Summarize the content of all text files in a directory.  
**Parameters:**  
- `directory` (str): Path to the directory.  
- `add_separator` (bool): Whether to add a separator between file contents.  
- `save_to_file` (bool): Whether to save the summarized content to a file.  
- `file_extension` (str, optional): File extension to operate on (e.g., `.txt`).  
- `summary_file_name` (str, optional): Name of the file to save the summary.  

---

### **BatchConvertImageFiles**  
**Function:** Convert all images in a specified directory to the target format.  
**Parameters:**  
- `directory` (str): Path to the directory.  
- `target_format` (str): Target image format (`jpg` or `png`).  
- `save_original` (bool): Whether to retain the original files after conversion.  

---

### **DedupImageFiles**  
**Function:** Remove duplicate image files in a specific directory.  
**Parameters:**  
- `directory` (str): Path to the directory.  
- `max_distance_threshold` (int): Maximum Hamming distance threshold for identifying duplicates.  

---

### **ShowAnythingMie**  
**Function:** Print the input content as a string.  
**Parameters:**  
- `anything` (*): The input content to display.  

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
