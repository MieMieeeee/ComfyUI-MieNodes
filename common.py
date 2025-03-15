import os
import json
import toml
from types import SimpleNamespace
from deepdiff import DeepDiff
from .utils import mie_log, any_typ

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
