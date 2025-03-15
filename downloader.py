import os
import re
import shutil
import requests
import urllib
from tqdm import tqdm

import folder_paths

from .utils import mie_log

MY_CATEGORY = "🐑 MieNodes/🐑 Downloader"


# Learned a lot from https://github.com/ciri/comfyui-model-downloader

class ModelDownloader(object):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": ""},),
                "save_path": ("STRING", {"default": "checkpoints"}),
                "override": ("BOOLEAN", {"default": True}),
                "use_hf_mirror": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "rename_to": ("STRING", {"default": ""},),
                "hf_token": ("STRING", {"default": "", "multiline": False, "password": True}),
                "trigger_signal": (("*", {})),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "download"

    CATEGORY = MY_CATEGORY

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    def download(self, url, save_path, override, use_hf_mirror, rename_to, hf_token, chunk_size=1024 * 1024, trigger_signal=None):
        if hf_token and "huggingface" in url:
            headers = {"Authorization": f"Bearer {hf_token}"}
        else:
            headers = None

        if use_hf_mirror:
           url = re.sub(r'^https://huggingface\.co', 'https://hf-mirror.com', url)

        response = requests.get(url, stream=True, headers=headers, params=None)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        file_name = rename_to
        if not file_name:
            file_name = self._get_filename(response, url)

        target_directory = os.path.join(folder_paths.models_dir, save_path)
        if not os.path.exists(target_directory):
            os.makedirs(target_directory, exist_ok=True)

        full_path = os.path.join(target_directory, file_name)

        if not override:
            if os.path.exists(full_path):
                return mie_log(f"File already exists and override is False: {full_path}"),

        temp_path = full_path + '.tmp'

        downloaded = 0
        try:
            with open(temp_path, 'wb') as file:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=file_name) as pbar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        size = file.write(data)
                        downloaded += size
                        pbar.update(size)

            # Only move the file if the download completed successfully
            shutil.move(temp_path, full_path)
            return mie_log(f"File {full_path} is downloaded from {url}"),

        except Exception as e:
            # Clean up the temp file if something goes wrong
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    @staticmethod
    def _get_filename(response, url):
        cd = response.headers.get('content-disposition')
        if cd:
            filenames = re.findall('filename="(.+?)"', cd)
            if filenames:
                return filenames[0]
        parsed_url = urllib.parse.urlparse(url)
        return os.path.basename(parsed_url.path)
