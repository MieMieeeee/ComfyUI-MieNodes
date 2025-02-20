import os
import json
import toml

from .utils import mie_log

MY_CATEGORY = "🐑 MieNodes/🐑 Common"


# Learned a lot from https://github.com/cubiq/ComfyUI_essentials

class ShowAnythingMie(object):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anything": (("*", {})),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    CATEGORY = MY_CATEGORY

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

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
                "data": (("*", {})),
                "directory": ("STRING", {"default": "X://path/to/folder"},),
                "file_name": ("STRING", {"default": "output"}),
                "save_format": (["txt", "json", "toml"], {}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_data"
    CATEGORY = MY_CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

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
