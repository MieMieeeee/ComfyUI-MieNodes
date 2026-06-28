import os
import json
import datetime
import logging
from logging.handlers import RotatingFileHandler
from threading import Lock
import toml
import glob
import hashlib
import time
import fnmatch
import shutil
from types import SimpleNamespace
from deepdiff import DeepDiff
import torch

import folder_paths
try:
    from _mienodes_internal.core.utils import mie_log, any_typ, compute_hash, convert_size
except ImportError:
    from ...core.utils import mie_log, any_typ, compute_hash, convert_size


# Empty IMAGE batch used as a safe fallback when LoadImageBatch|Mie
# cannot find the requested file and no upstream fallback is connected.
EMPTY_IMAGE_BATCH = torch.zeros((0, 1, 1, 3), dtype=torch.float32)

def _plugin_root_dir():
    """Return the on-disk directory of the installed plugin.

    ``general.py`` lives at ``<plugin>/nodes/common/general.py``; walking up
    three levels reaches the plugin root no matter how the user installed
    it (clone, symlink, embedded venv, etc.).
    """
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def _safe_repr(value, limit=2000):
    """Return ``str(value)`` clamped to ``limit`` characters."""
    s = "" if value is None else str(value)
    if len(s) > limit:
        s = s[:limit] + "... <truncated %d chars>" % (len(s) - limit)
    return s


def _extract_upstream(extra_pnginfo, unique_id):
    """Best-effort lookup of the upstream node connected to ``anything``.

    Returns a dict with optional keys: ``node_id``, ``node_title``,
    ``node_type``, ``upstream_id``, ``upstream_title``, ``upstream_type``.
    Missing values are simply absent from the dict.
    """
    info = {}
    if not extra_pnginfo or not isinstance(extra_pnginfo, dict):
        return info
    workflow = extra_pnginfo.get("workflow")
    if not isinstance(workflow, dict):
        return info
    nodes = workflow.get("nodes") or []
    links = workflow.get("links") or []
    if not isinstance(nodes, list) or not isinstance(links, list):
        return info
    me = None
    for n in nodes:
        if isinstance(n, dict) and str(n.get("id")) == str(unique_id):
            me = n
            break
    if me is None:
        return info
    info["node_id"] = me.get("id")
    info["node_title"] = me.get("title")
    info["node_type"] = me.get("type")
    # locate the ``anything`` input connection (declared first on this node)
    inputs = me.get("inputs") or []
    target = None
    for inp in inputs:
        if isinstance(inp, dict) and inp.get("name") == "anything":
            target = inp
            break
    if target is None:
        for inp in inputs:
            if isinstance(inp, dict) and inp.get("link") is not None:
                target = inp
                break
    if target is None:
        return info
    link = target.get("link")
    if isinstance(link, list) and link:
        link = link[0]
    if link is None:
        return info
    for entry in links:
        if not (isinstance(entry, list) and len(entry) >= 4):
            continue
        if entry[0] != link:
            continue
        src_id = entry[1]
        for sn in nodes:
            if isinstance(sn, dict) and sn.get("id") == src_id:
                info["upstream_id"] = src_id
                info["upstream_title"] = sn.get("title")
                info["upstream_type"] = sn.get("type")
                return info
    return info


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


class ShowAndSaveAnythingMie(object):
    """Like ``ShowAnythingMie`` but also appends a structured log line.

    The log entry is JSON-Lines and captures timestamp, the current node's
    id / title / type, the upstream node wired into ``anything`` (id /
    title / type), and
    the result. Files rotate on size and live in ``<plugin>/logs/``; the
    user picks the file name.
    """

    LOG_DIR_NAME = "logs"
    LOG_DEFAULT_NAME = "show_anything.log"
    LOG_MAX_BYTES = 25 * 1024 * 1024  # 25 MB per file (4 files total = 100 MB cap)
    LOG_BACKUP_COUNT = 3

    _LOGGER_CACHE = {}
    _LOGGER_LOCK = Lock()

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anything": (any_typ,),
                "save_to_log": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "log_file_name": ("STRING", {"default": "show_anything.log"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    CATEGORY = MY_CATEGORY

    def execute(self, anything, save_to_log=True, log_file_name="",
                unique_id=None, extra_pnginfo=None):
        """
        以字符形式打印输入的内容，同时将时间、上游节点信息与结果写入日志文件。

        参数：
        - anything (*): 输入的内容
        - save_to_log (bool): 是否写入日志（默认 True）
        - log_file_name (str): 日志文件名（仅文件名，不含路径），留空使用 show_anything.log
        """

        text = str(anything)
        mie_log(f"ShowAndSaveAnythingMie: {text}")

        if save_to_log:
            self._write_log_entry(
                anything, text, log_file_name, unique_id, extra_pnginfo
            )

        return {"ui": {"text": text}, "result": (text,)}

    @classmethod
    def _plugin_log_dir(cls):
        log_dir = os.path.join(_plugin_root_dir(), cls.LOG_DIR_NAME)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    @classmethod
    def _resolve_log_path(cls, log_file_name):
        """Resolve a log file path inside the plugin's ``logs/`` folder.

        The user supplies a *file name* (no directory). Empty / whitespace
        falls back to ``show_anything.log``. Any directory components the
        user may have included are stripped via ``os.path.basename`` to
        keep the result inside the plugin's ``logs/`` directory.
        """
        name = (log_file_name or "").strip() or cls.LOG_DEFAULT_NAME
        name = os.path.basename(name)
        return os.path.join(cls._plugin_log_dir(), name)

    @classmethod
    def _get_logger(cls, log_file_path):
        """Return a cached logger that size-rotates ``log_file_path``."""
        with cls._LOGGER_LOCK:
            cached = cls._LOGGER_CACHE.get(log_file_path)
            if cached is not None:
                return cached
            os.makedirs(os.path.dirname(log_file_path) or ".", exist_ok=True)
            logger = logging.getLogger("MieShowAndSave:" + log_file_path)
            logger.setLevel(logging.INFO)
            logger.propagate = False
            for h in list(logger.handlers):
                logger.removeHandler(h)
            handler = RotatingFileHandler(
                log_file_path,
                maxBytes=cls.LOG_MAX_BYTES,
                backupCount=cls.LOG_BACKUP_COUNT,
                encoding="utf-8",
                delay=True,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            cls._LOGGER_CACHE[log_file_path] = logger
            return logger

    @classmethod
    def _write_log_entry(cls, anything, text, log_file_name,
                         unique_id, extra_pnginfo):
        try:
            log_path = cls._resolve_log_path(log_file_name)
            meta = _extract_upstream(extra_pnginfo, unique_id)
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            upstream_title = (
                meta.get("upstream_title")
                or meta.get("upstream_type")
                or "(unknown)"
            )
            entry = {
                "ts": ts,
                "upstream_title": str(upstream_title),
                "result": _safe_repr(text),
            }
            logger = cls._get_logger(log_path)
            logger.info(json.dumps(entry, ensure_ascii=False))
            mie_log(f"ShowAndSaveAnythingMie: log written to {log_path}")
        except Exception as e:
            mie_log(f"ShowAndSaveAnythingMie: failed to write log: {e}")


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


class GetFileBasename(object):
    """Return the basename of a file path, with the file extension stripped.

    Use this when a cache key or output filename needs to be derived from
    an input file (e.g. drive video, ref image). The output is a plain
    STRING so it can be spliced into StringConcat|Mie chains.

    Inputs:
    - file_path: a STRING path (absolute or relative to ComfyUI's
      base_path). Non-string input falls back to "".

    Outputs:
    - basename: the final path component, e.g. "video" for
      "input/video.mp4" or "/abs/path/video.mp4".

    This is a control-plane utility only: no disk I/O, no hashing, no
    side effect beyond computing the string. It does NOT enforce that
    the file exists; the caller can run FileExists|Mie after.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "input/foo.mp4"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("basename",)
    FUNCTION = "execute"
    CATEGORY = "🐑 MieNodes/🐑 Common"

    def execute(self, file_path):
        if not isinstance(file_path, str):
            return ("",)
        p = file_path.strip()
        if not p:
            return ("",)
        try:
            base = os.path.basename(p)
        except Exception:
            return ("",)
        stem, _ext = os.path.splitext(base)
        return (stem,)


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


class FileExists(object):
    """Lightweight existence check for a single file path.

    Intentionally minimal: a single ``os.path.isfile()`` call, no size /
    hash / mtime lookup (use ``GetFileInfo|Mie`` for those). The path is
    taken as-is (absolute or relative to ComfyUI CWD); pair this node with
    ``GetAbsolutePath|Mie`` to resolve a workflow-relative path first.

    Empty / whitespace input is treated as not-exists rather than raising,
    so the node is safe to wire into a cache-gate before any path has been
    materialised.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "output/cache/abc.mp4"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("exists",)
    FUNCTION = "execute"
    CATEGORY = "🐑 MieNodes/🐑 Common"

    def execute(self, file_path):
        raw = str(file_path or "").strip()
        if not raw:
            return (False,)
        result = os.path.isfile(raw)
        if raw:
            mie_log(f"FileExists|Mie: {raw} -> {result}")
        return (result,)


class IfElse(object):
    """Generic if/else router driven by an external BOOLEAN.

    Differs from ``MieLoopIfIsLast` (which only knows the loop's
    ``is_last`` flag) and ``MieLoopIfCurrentIdx`` (which only compares
    the loop index) -- this node takes any boolean computed anywhere
    upstream and uses it to pick a branch. The boolean is typically the
    output of ``FileExists|Mie`` (cache hit/miss), but can be any other
    condition in the graph.

    ``then_value`` and ``else_value`` are ``any_typ`` so any data type
    (image, video, string, dict, loop_ctx, ...) can flow through.
    Unconnected branches fall back to ``None``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"forceInput": True}),
            },
            "optional": {
                "then_value": (any_typ,),
                "else_value": (any_typ,),
            },
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "🐑 MieNodes/🐑 Common"

    def execute(self, condition, then_value=None, else_value=None):
        chosen = then_value if bool(condition) else else_value
        return (chosen,)
class SaveImageBatch(object):
    """Persist a ComfyUI IMAGE batch (N,H,W,3 float32 in [0,1]) as a lossless .pt file.

    The path is taken as-is (absolute or ComfyUI-CWD relative). Parent directories are created on demand. This is the cache write side used by the SCAIL-2 material-cache loop: the 81-frame SCAIL-2 material batch is the expensive part of the workflow, so we materialise it once and reload it on subsequent runs of the same source video.

    .pt is preferred over .mp4 here because the data is exactly the ComfyUI IMAGE tensor (no encode/decode round-trip), the file path is deterministic (VHS auto-numbers by fps/counter), and the on-disk size stays close to N*H*W*3*4 bytes.
    """

    @classmethod
    def VALIDATE_INPUTS(s, images, file_path):
        return True
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "file_path": ("STRING", {"default": "output/cache/foo.pt"}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "🐑 MieNodes/🐑 Common"

    def execute(self, images, file_path):
        raw = str(file_path or "").strip()
        if not raw:
            raise ValueError("SaveImageBatch|Mie: file_path is empty")
        parent = os.path.dirname(raw)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        payload = images.detach().to("cpu").contiguous()
        torch.save(payload, raw)
        try:
            size = os.path.getsize(raw)
        except OSError:
            size = -1
        mie_log(f"SaveImageBatch|Mie: wrote {raw} ({size} bytes)")
        return ()

class LoadImageBatch(object):
    """Inverse of SaveImageBatch|Mie: reload a .pt IMAGE batch.

    Designed for a cache gate: when file_path is missing or empty, the node falls back to the optional fallback IMAGE input (typically the freshly-generated material batch). When fallback is also unconnected, an empty batch is returned so the downstream graph still receives a valid IMAGE and can decide what to do.

    Wire it after FileExists|Mie only if the consumer needs a separate existence signal -- this node already does the existence check internally, so most call sites do not need an IfElse wrapper.
    """

    @classmethod
    def VALIDATE_INPUTS(s, file_path, fallback=None):
        return True
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "output/cache/foo.pt"}),
            },
            "optional": {
                "fallback": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "🐑 MieNodes/🐑 Common"

    def execute(self, file_path, fallback=None):
        raw = str(file_path or "").strip()
        if raw and os.path.isfile(raw):
            try:
                size = os.path.getsize(raw)
            except OSError:
                size = -1
            mie_log(f"LoadImageBatch|Mie: cache HIT loaded {raw} ({size} bytes)")
            return (torch.load(raw, map_location="cpu"),)
        mie_log(f"LoadImageBatch|Mie: cache MISS {raw} -> fallback")
        if fallback is not None:
            return (fallback,)
        return (EMPTY_IMAGE_BATCH,)


class SaveAny(object):
    """Persist any Python object via pickle to a deterministic file path.

    Generic sibling of SaveImageBatch|Mie: instead of torch.save on a ComfyUI IMAGE
    tensor, this uses plain pickle so callers can round-trip arbitrary data
    (dicts, custom plugin outputs like SAM3_TRACK_DATA, lists, etc.) through
    the same file-gate idiom. Use this when the value type does not match
    SaveImageBatch's IMAGE contract -- e.g. the SAM3 drive/ref track dict that
    feeds SCAIL2ColoredMask.

    Pair with LoadAny|Mie + FileExists|Mie + IfElse|Mie to build a cache gate
    identical in shape to the 351[0]/351[1] IMAGE cache (see v5 workflow).
    """

    @classmethod
    def VALIDATE_INPUTS(s, value, file_path):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (any_typ,),
                "file_path": ("STRING", {"default": "output/cache/foo.pkl"}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "🐑 MieNodes/🐑 Common"

    def execute(self, value, file_path):
        import pickle
        raw = str(file_path or "").strip()
        if not raw:
            raise ValueError("SaveAny|Mie: file_path is empty")
        parent = os.path.dirname(raw)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(raw, "wb") as f:
            pickle.dump(value, f)
        try:
            size = os.path.getsize(raw)
        except OSError:
            size = -1
        mie_log(f"SaveAny|Mie: wrote {raw} ({size} bytes)")
        return ()


class LoadAny(object):
    """Inverse of SaveAny|Mie: unpickle any Python object from a file path.

    Mirrors LoadImageBatch|Mie's contract: when the file is missing or empty,
    fall back to the optional wildcard fallback input. When fallback is also
    unconnected, return None so the downstream graph still receives a valid
    (None) value and can decide what to do. Pair with FileExists|Mie only if
    the consumer needs a separate existence signal -- this node already does
    the existence check internally.
    """

    @classmethod
    def VALIDATE_INPUTS(s, file_path, fallback=None):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "output/cache/foo.pkl"}),
            },
            "optional": {
                "fallback": (any_typ,),
            },
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "🐑 MieNodes/🐑 Common"

    def execute(self, file_path, fallback=None):
        import pickle
        raw = str(file_path or "").strip()
        if raw and os.path.isfile(raw):
            try:
                size = os.path.getsize(raw)
            except OSError:
                size = -1
            mie_log(f"LoadAny|Mie: cache HIT loaded {raw} ({size} bytes)")
            with open(raw, "rb") as f:
                return (pickle.load(f),)
        mie_log(f"LoadAny|Mie: cache MISS {raw} -> fallback")
        if fallback is not None:
            return (fallback,)
        return (None,)


class LoadOrCompute(object):
    """Disk cache gate that can skip its own upstream node on a cache hit.

    This is the load-or-compute-save pattern expressed as a single ComfyUI
    node, built on ComfyUI's official Lazy Evaluation mechanism. It replaces
    the ``LoadAny + SaveAny + IfElse`` trio (and the old MieLoopCacheMarker
    graph-rewrite approach): the computation upstream of ``value`` is gated
    purely at runtime via ``check_lazy_status``, not by editing the graph.

    Contract:
      - ``cache_path`` is a STRING resolved eagerly (must be known before
        ``check_lazy_status`` is called). Per T22's cache-key design it is
        derived from inputs that live upstream of the heavy target
        (video basename / size / mode / frame range), so it never depends on
        the lazy ``value`` itself.
      - ``value`` is ``any_typ`` and declared ``lazy=True``. The engine does
        NOT compute it up-front; instead it calls ``check_lazy_status`` first.
      - HIT (file exists): ``check_lazy_status`` returns ``[]`` -> the engine
        skips the upstream node entirely -> ``execute`` loads and returns the
        cached payload. This is the real win: the expensive upstream (e.g.
        SAM3_VideoTrack) does not run on a cache hit.
      - MISS (file absent): ``check_lazy_status`` returns ``["value"]`` -> the
        engine runs the upstream node -> ``execute`` receives the freshly
        computed value, persists it to ``cache_path``, and returns it.

    Design decisions (aligned with user requirements):
      - Empty/whitespace ``cache_path`` raises (option B): there is no
        legitimate reason to wire this node without a cache key. Silent
        passthrough would mask wiring bugs.
      - Serialisation mirrors SaveAny/LoadAny: pickle for arbitrary Python
        objects (SAM3TrackData dicts, etc.). IMAGE batches should be cached
        with SaveImageBatch/LoadImageBatch for the lossless .pt path; this
        node is the general-purpose pickle variant.
      - Single responsibility: it does NOT know what the upstream computes.
        It is a generic gate for any lazy input.

    See: ComfyUI Lazy Evaluation docs and ``comfy_extras/nodes_logic.py``
    ``SwitchNode`` for the authoritative ``check_lazy_status`` contract.
    """

    @classmethod
    def VALIDATE_INPUTS(s, cache_path, value=None):
        # Mirror the no-op validator used by SaveAny/LoadAny: this node's own
        # inputs are always structurally valid (cache_path is a STRING widget;
        # value is a lazy link or unconnected). Returning True here lets the
        # v3 recursive validator resolve cleanly without delegating to the
        # upstream node's VALIDATE_INPUTS with a mismatched input key.
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_path": ("STRING", {"default": "output/cache/foo.pkl"}),
                "value": (any_typ, {"lazy": True}),
            },
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "🐑 MieNodes/🐑 Common"

    def check_lazy_status(self, cache_path, value=None):
        # The engine calls this before evaluating the lazy ``value`` input.
        # Returning [] means "do not compute value" -> upstream node is skipped.
        # cache_path is already resolved here (it is a non-lazy input), so the
        # existence check is safe and deterministic.
        raw = str(cache_path or "").strip()
        if raw and os.path.isfile(raw):
            mie_log(f"LoadOrCompute|Mie: 找到缓存 {raw} -> 跳过上游计算 (cache HIT, suppress upstream)")
            return []
        mie_log(f"LoadOrCompute|Mie: 没有缓存 {raw} -> 需要上游计算 (cache MISS, request upstream)")
        return ["value"]

    def execute(self, cache_path, value=None):
        import pickle
        raw = str(cache_path or "").strip()
        if not raw:
            # Option B: no cache key is a wiring error, never silent.
            raise ValueError(
                "LoadOrCompute|Mie: cache_path is empty -- a cache key is required"
            )
        if os.path.isfile(raw):
            try:
                size = os.path.getsize(raw)
            except OSError:
                size = -1
            mie_log(f"LoadOrCompute|Mie: 找到缓存 已加载 {raw} ({size} bytes) (cache HIT loaded)")
            with open(raw, "rb") as f:
                return (pickle.load(f),)
        # MISS path: value is the freshly computed upstream output.
        if value is None:
            raise ValueError(
                f"LoadOrCompute|Mie: 没有缓存但上游返回 None {raw} (cache MISS, upstream None)"
            )
        parent = os.path.dirname(raw)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(raw, "wb") as f:
            pickle.dump(value, f)
        try:
            size = os.path.getsize(raw)
        except OSError:
            size = -1
        mie_log(f"LoadOrCompute|Mie: 保存缓存 {raw} ({size} bytes) (cache MISS saved)")
        return (value,)


class ImageHash(object):
    """Deterministic content hash of a ComfyUI IMAGE batch, returned as a STRING.

    Solves the cache-key problem for inputs whose filename is not exposed as a
    STRING output -- most notably ``LoadImage`` (its ``image`` widget holds the
    filename, but only IMAGE/MASK come out) and ``VHS_LoadVideo`` (its ``video``
    widget is hidden behind the VHS_VIDEOINFO output). By hashing the IMAGE
    tensor we get a stable key that tracks the actual content the user selected,
    with no manual filename sync and no graph-walk to read upstream widgets.

    Usage: wire it directly downstream of the IMAGE source (LoadImage for a
    reference image, VHS_LoadVideo for a drive video) and splice the resulting
    short hash into a StringConcat|Mie cache-key chain. One node, fully
    automatic: the user only ever touches the LoadImage / VHS widgets.

    ``frame_limit`` controls how much of the batch is hashed:
      - 0  : hash the entire batch (use for reference images -- 1 frame, <10ms)
      - >0 : hash only the first N frames (use for drive video identity).
             A drive video is identified by hashing its first frame only
             (frame_limit=1): two different videos differ on frame 0 almost
             surely, so this is a reliable identity key at negligible cost
             (~10ms regardless of video length). A full 81-frame hash would
             cost ~100ms+ and add nothing for identity purposes.

    Determinism:
      - Input is detached + moved to CPU + made contiguous before hashing, so a
        GPU tensor and a CPU tensor with identical values hash equally.
      - Uses sha256 over the raw float32 bytes; truncated to 12 hex chars (48
        bits) -- collision-safe for a per-user cache directory.
    """

    @classmethod
    def VALIDATE_INPUTS(s, images, frame_limit=0):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "frame_limit": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hash",)
    FUNCTION = "execute"
    CATEGORY = "🐑 MieNodes/🐑 Common"

    def execute(self, images, frame_limit=0):
        try:
            t = images.detach().to("cpu").contiguous()
            if frame_limit and frame_limit > 0 and t.shape[0] > frame_limit:
                t = t[:frame_limit]
            buf = t.numpy().tobytes() if hasattr(t, "numpy") else bytes(t)
            h = hashlib.sha256(buf).hexdigest()[:12]
            return (h,)
        except Exception as e:
            mie_log(f"ImageHash|Mie: hash failed ({e}), falling back to empty")
            return ("",)

