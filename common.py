import os
import json
import toml
import glob
import hashlib
import time
import fnmatch
import shutil
from types import SimpleNamespace
from deepdiff import DeepDiff

import folder_paths
from .utils import mie_log, any_typ, compute_hash, convert_size

MY_CATEGORY = "🐑 MieNodes/🐑 Common"


# Learned a lot from https://github.com/cubiq/ComfyUI_essentials

class ShowAnythingMie(object):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anything": (any_typ,),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    CATEGORY = MY_CATEGORY

    def execute(self, anything):
        """
        以字符形式打印输入的内容。

        参数：
        - input (*): 输入的内容

        返回：
        - 给UI的json格式
        """

        text = str(anything)
        mie_log(f"ShowAnythingMie: {text}")

        return {"ui": {"text": text}, "result": (text,)}


class SaveAnythingAsFile(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (any_typ,),
                "directory": ("STRING", {"default": "X://path/to/folder"},),
                "file_name": ("STRING", {"default": "output"}),
                "save_format": (["txt", "json", "toml"], {}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_data"
    CATEGORY = MY_CATEGORY
    OUTPUT_NODE = True

    def save_data(self, data, directory, file_name, save_format):
        """
        Save data to a file in either TOML, JSON, or TXT format.

        Parameters:
        - data (*): The data to save
        - directory (str): The directory to save the file in
        - file_name (str): The name of the output file
        - format (str): The format to save the data in ("json", "toml", or "txt")

        Returns:
        - A message indicating success or failure
        """
        file_path = os.path.join(directory, f"{file_name}.{save_format}")
        try:
            if save_format == "json":
                try:
                    json_data = json.dumps(data, indent=4)
                except (TypeError, ValueError) as e:
                    return mie_log(f"Failed to serialize data to JSON: {e}")
                with open(file_path, 'w') as f:
                    f.write(json_data)
            elif save_format == "toml":
                try:
                    if isinstance(data, SimpleNamespace):
                        data = vars(data)
                    elif isinstance(data, dict):
                        data = data
                    else:
                        data = data.__dict__
                    toml_data = toml.dumps(data)
                except (TypeError, ValueError) as e:
                    return mie_log(f"Failed to serialize data to TOML: {e}")
                with open(file_path, 'w') as f:
                    f.write(toml_data)
            elif save_format == "txt":
                with open(file_path, 'w') as f:
                    if isinstance(data, str):
                        f.write(data)
                    else:
                        f.write(str(data))
            else:
                return mie_log("Unsupported format. Please choose 'json', 'toml', or 'txt'.")
            return mie_log(f"Data successfully saved to {file_path} in {save_format} format.")
        except Exception as e:
            return mie_log(f"Failed to save data: {e}")


class CompareFiles(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file1_path": ("STRING", {"default": "X://path/to/file1"}),
                "file2_path": ("STRING", {"default": "X://path/to/file2"}),
                "file_format": (["json", "toml"], {}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "compare_files"
    CATEGORY = "🐑 MieNodes/🐑 Common"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    def convert_sets_to_lists(self, data):
        if isinstance(data, set):
            return list(data)
        elif isinstance(data, dict):
            return {k: self.convert_sets_to_lists(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_sets_to_lists(i) for i in data]
        else:
            return data

    def compare_files(self, file1_path, file2_path, file_format):
        """
        Compare two files and return the differences.

        Parameters:
        - file1_path (str): The path to the first file
        - file2_path (str): The path to the second file
        - file_format (str): The format of the files ("json" or "toml")

        Returns:
        - A string describing the differences
        """
        try:
            if file_format == "json":
                with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
                    data1 = json.load(f1)
                    data2 = json.load(f2)
            elif file_format == "toml":
                with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
                    data1 = toml.load(f1)
                    data2 = toml.load(f2)
            else:
                return mie_log("Unsupported format. Please choose 'json' or 'toml'."),

            data1 = self.convert_sets_to_lists(data1)
            data2 = self.convert_sets_to_lists(data2)

            differences = DeepDiff(data1, data2, ignore_order=True, report_repetition=True).to_dict()
            formatted_diff = self.format_diff(differences, file1_path, file2_path)
            return formatted_diff,
        except Exception as e:
            return mie_log(f"Failed to compare files: {e}"),

    def format_diff(self, differences, file1_name, file2_name):
        """
        Format the DeepDiff output to key:\n\tfile1: value1\n\tfile2: value2 format.

        Parameters:
        - differences (dict): The DeepDiff output
        - file1_name (str): The name of the first file
        - file2_name (str): The name of the second file

        Returns:
        - A formatted string
        """
        formatted = []
        for change_type, changes in differences.items():
            if isinstance(changes, dict):
                for key, change in changes.items():
                    short_key = key.split('[')[-1].strip("']")
                    if change_type == 'values_changed':
                        old_value = change['old_value']
                        new_value = change['new_value']
                    elif change_type == 'dictionary_item_added':
                        old_value = 'null'
                        new_value = change
                    elif change_type == 'dictionary_item_removed':
                        old_value = change
                        new_value = 'null'
                    else:
                        continue
                    formatted.append(f"{short_key}:\n\t{file1_name}: {old_value}\n\t{file2_name}: {new_value}")
            elif isinstance(changes, set):
                for change in changes:
                    short_key = change.split('[')[-1].strip("']")
                    if change_type == 'dictionary_item_added':
                        formatted.append(f"{short_key}:\n\t{file1_name}: null\n\t{file2_name}: {changes[change]}")
                    elif change_type == 'dictionary_item_removed':
                        formatted.append(f"{short_key}:\n\t{file1_name}: {changes[change]}\n\t{file2_name}: null")
        return "\n".join(formatted)


class GetAbsolutePath(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "relative_path": ("STRING", {"default": "input/abc"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("absolute_path",)
    FUNCTION = "execute"
    CATEGORY = "🐑 MieNodes/🐑 Common"

    def execute(self, relative_path):
        return os.path.join(folder_paths.base_path, relative_path),


class GetFileInfo(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "input/abc"}),
                "hash_algorithm": (["md5", "sha1", "sha256", "None"], {"default": "sha256"}),
            },
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("file_info",)
    FUNCTION = "execute"
    CATEGORY = "🐑 MieNodes/🐑 Common"

    def execute(self, file_path, hash_algorithm):
        file_path = os.path.join(folder_paths.base_path, file_path)

        file_info = {}
        try:
            file_info['size'] = convert_size(os.path.getsize(file_path)),
            file_info['creation_time'] = time.ctime(os.path.getctime(file_path))
            file_info['modification_time'] = time.ctime(os.path.getmtime(file_path))
            file_info['hash'] = compute_hash(file_path, hash_algorithm)
            file_info_json = json.dumps(file_info, indent=4)
        except Exception as e:
            return json.dumps({"error": str(e)}),
        return file_info_json,

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


class GetDirectoryFilesInfo(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "input/abc"}),
                "hash_algorithm": (["md5", "sha1", "sha256", "None"], {"default": "None"}),
                "pattern": ("STRING", {"default": "*"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("directory_files_info",)
    FUNCTION = "execute"
    CATEGORY = "🐑 MieNodes/🐑 Common"

    def execute(self, directory_path, hash_algorithm, pattern):
        directory_path = os.path.join(folder_paths.base_path, directory_path)
        pattern = "*" if not pattern else pattern

        directory_files_info = []
        try:
            for root, _, files in os.walk(directory_path):
                for file in fnmatch.filter(files, pattern):
                    file_path = os.path.join(root, file)
                    file_info = {
                        'file_path': file_path,
                        'size': convert_size(os.path.getsize(file_path)),
                        'creation_time': time.ctime(os.path.getctime(file_path)),
                        'modification_time': time.ctime(os.path.getmtime(file_path)),
                        'hash': compute_hash(file_path, hash_algorithm)
                    }
                    directory_files_info.append(file_info)
            directory_files_info_json = json.dumps(directory_files_info, indent=4)
        except Exception as e:
            return json.dumps({"error": str(e)}),
        return directory_files_info_json,

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


class CopyFiles(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_directory": ("STRING", {"default": "X://path/to/source"}),
                "destination_directory": ("STRING", {"default": "X://path/to/destination"}),
                "pattern": ("STRING", {"default": "*.txt"}),
                "delete_origin": ("BOOLEAN", {"default": False}),
                "mock": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("copied_file_count", "log")
    FUNCTION = "copy_files"
    CATEGORY = "🐑 MieNodes/🐑 File Operations"

    def copy_files(self, source_directory, destination_directory, pattern, delete_origin, mock):
        """
        Copy files matching a pattern from source to destination directory.

        Parameters:
        - source_directory (str): Path to the source directory.
        - destination_directory (str): Path to the destination directory.
        - pattern (str): File pattern to match (e.g., "*.txt").
        - delete_origin (bool): Whether to delete the original files after copying.
        - mock (bool): If True, only log the actions without performing them.

        Returns:
        - Number of files copied.
        - Log message.
        """

        source_directory = os.path.join(folder_paths.base_path, source_directory)
        destination_directory = os.path.join(folder_paths.base_path, destination_directory)
        pattern = "*" if not pattern else pattern

        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory, exist_ok=True)

        copied_count = 0
        log_messages = []
        try:
            for root, _, files in os.walk(source_directory):
                for file in fnmatch.filter(files, pattern):
                    source_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, source_directory)
                    destination_path = os.path.join(destination_directory, relative_path, file)

                    if not mock:
                        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                        shutil.copy2(source_path, destination_path)
                        log_messages.append(f"Copied {source_path} to {destination_path}")
                        copied_count += 1

                        if delete_origin:
                            os.remove(source_path)
                            log_messages.append(f"Deleted {source_path}")
                    else:
                        log_messages.append(f"[MOCK] Would copy from: {source_path} to: {destination_path}")
                        copied_count += 1
                        if delete_origin:
                            log_messages.append(f"[MOCK] Would delete source file: {source_path}")

            if not mock:
                log_messages.append(
                    f"Copied {copied_count} files matching '{pattern}' from {source_directory} to {destination_directory}.")
            elif copied_count == 0:
                log_messages.append(f"No files matching '{pattern}' found in {source_directory}.")
            return copied_count, mie_log("\n".join(log_messages))
        except Exception as e:
            return 0, mie_log(f"Failed to copy files: {e}")


class DeleteFiles(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "X://path/to/file_or_directory/*.txt"}),  # Supports patterns
                "mock": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "delete_files"
    CATEGORY = "🐑 MieNodes/🐑 File Operations"

    def delete_files(self, path, mock):
        """
        Delete files or directories matching a pattern.

        Parameters:
        - path (str): Path or pattern to match files or directories (e.g., "aa/cc/*.txt").
        - mock (bool): If True, only log the actions without performing them.

        Returns:
        - Log message.
        """
        path = os.path.join(folder_paths.base_path, path)
        log_messages = []

        try:
            matched_paths = glob.glob(path, recursive=True)
            if not matched_paths:
                log_messages.append(
                    f"[MOCK] No matching files or directories for: {path}" if mock else f"No matching files or directories for: {path}")
            for matched_path in matched_paths:
                if os.path.isfile(matched_path):
                    if mock:
                        log_messages.append(f"[MOCK] Would delete file: {matched_path}")
                    else:
                        os.remove(matched_path)
                        log_messages.append(f"Deleted file: {matched_path}")
                elif os.path.isdir(matched_path):
                    if mock:
                        log_messages.append(f"[MOCK] Would delete directory and its contents: {matched_path}")
                    else:
                        shutil.rmtree(matched_path)
                        log_messages.append(f"Deleted directory and its contents: {matched_path}")
        except Exception as e:
            log_messages.append(f"Failed to delete {path}: {e}")

        return mie_log("\n".join(log_messages)),


class ClassicAspectRatio(object):
    @classmethod
    def INPUT_TYPES(cls):
        resolutions_by_ratio = {
            "1:1": [
                "512x512 ( 0.25MP )", "768x768 ( 0.56MP )", "1024x1024 ( 1MP )", "1280x1280 ( 1.56MP )", "1536x1536 ( 2.25MP )", "2048x2048 ( 4MP )",
            ],
            "2:3": [
                "640x960 ( 0.59MP )", "768x1152 ( 0.84MP )", "832x1248 ( 0.99MP )", "1024x1536 ( 1.5MP )", "1248x1872 ( 2.23MP )", "1664x2496 ( 3.96MP )",
            ],
            "3:2": [
                "960x640 ( 0.59MP )", "1152x768 ( 0.84MP )", "1248x832 ( 0.99MP )", "1536x1024 ( 1.5MP )", "1872x1248 ( 2.23MP )", "2496x1664 ( 3.96MP )",
            ],
            "3:4": [
                "480x640 ( 0.29MP )", "720x960 ( 0.66MP )", "864x1152 ( 0.95MP )", "1104x1472 ( 1.55MP )", "1296x1728 ( 2.14MP )", "1728x2304 ( 3.8MP )",
            ],
            "4:3": [
                "640x480 ( 0.29MP )", "960x720 ( 0.66MP )", "1152x864 ( 0.95MP )", "1472x1104 ( 1.55MP )", "1728x1296 ( 2.14MP )", "2304x1728 ( 3.8MP )",
            ],
            "7:9": [
                "448x576 ( 0.25MP )", "560x720 ( 0.38MP )", "896x1152 ( 0.98MP )", "1120x1440 ( 1.54MP )", "1344x1728 ( 2.21MP )", "1792x2304 ( 3.94MP )",
            ],
            "9:7": [
                "576x448 ( 0.25MP )", "720x560 ( 0.38MP )", "1152x896 ( 0.98MP )", "1440x1120 ( 1.54MP )", "1728x1344 ( 2.21MP )", "2304x1792 ( 3.94MP )",
            ],
            "9:16": [
                "432x768 ( 0.32MP )", "576x1024 ( 0.56MP )", "720x1280 ( 0.88MP )", "864x1536 ( 1.27MP )", "1152x2048 ( 2.25MP )", "1512x2688 ( 3.88MP )",
            ],
            "16:9": [
                "768x432 ( 0.32MP )", "1024x576 ( 0.56MP )", "1280x720 ( 0.88MP )", "1536x864 ( 1.27MP )", "2048x1152 ( 2.25MP )", "2688x1512 ( 3.88MP )",
            ],
            "9:21": [
                "384x896 ( 0.33MP )", "432x1008 ( 0.42MP )", "576x1344 ( 0.74MP )", "720x1680 ( 1.15MP )", "864x2016 ( 1.66MP )", "1312x3072 ( 3.84MP )",
            ],
            "21:9": [
                "896x384 ( 0.33MP )", "1008x432 ( 0.42MP )", "1344x576 ( 0.74MP )", "1680x720 ( 1.15MP )", "2016x864 ( 1.66MP )", "3072x1312 ( 3.84MP )",
            ],
        }

        return {
            "required": {
                "ratio": ([
                    "1:1", "2:3", "3:2", "3:4", "4:3", "7:9", "9:7", "9:16", "16:9", "9:21", "21:9"
                ], {"default": "1:1"}),
                "resolution": (resolutions_by_ratio["1:1"], {"default": "1024x1024 ( 1MP )"}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_size"
    CATEGORY = "🐑 MieNodes/🐑 Common"

    def get_size(self, ratio, resolution):
        try:
            size_part = resolution.split(" (")[0]
            w_str, h_str = size_part.split("x")
            w = int(w_str.strip())
            h = int(h_str.strip())
            return w, h
        except Exception:
            return 1024, 1024
