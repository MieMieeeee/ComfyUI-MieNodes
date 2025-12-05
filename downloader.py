import os
import re
import shutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
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
                "skip_ssl_verify": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {"default": 30, "min": 1, "max": 600}),
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

    def download(self, url, save_path, override, use_hf_mirror, rename_to, hf_token, chunk_size=1024 * 1024, skip_ssl_verify=True, timeout=30, trigger_signal=None):
        if hf_token and "huggingface" in url:
            headers = {"Authorization": f"Bearer {hf_token}"}
        else:
            headers = None

        if use_hf_mirror:
           url = re.sub(r'^https://huggingface\.co', 'https://hf-mirror.com', url)

        verify = not skip_ssl_verify

        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"]) 
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        target_directory = os.path.join(folder_paths.models_dir, save_path)
        if not os.path.exists(target_directory):
            os.makedirs(target_directory, exist_ok=True)

        file_name = rename_to
        if not file_name:
            # preflight name by requesting headers only when needed
            head_resp = session.get(url, stream=True, headers=headers, params=None, verify=verify, timeout=timeout)
            head_resp.raise_for_status()
            file_name = self._get_filename(head_resp, url)

        full_path = os.path.join(target_directory, file_name)

        if not override:
            if os.path.exists(full_path):
                return mie_log(f"File already exists and override is False: {full_path}"),

        temp_path = full_path + '.tmp'

        downloaded = 0
        if os.path.exists(temp_path):
            downloaded = os.path.getsize(temp_path)

        req_headers = headers or {}
        if downloaded > 0:
            req_headers = dict(req_headers)
            req_headers["Range"] = f"bytes={downloaded}-"

        response = session.get(url, stream=True, headers=req_headers, params=None, verify=verify, timeout=timeout)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        cr = response.headers.get('content-range')
        if cr and '/' in cr:
            try:
                total_size = int(cr.split('/')[-1])
            except Exception:
                total_size = downloaded + int(response.headers.get('content-length', 0))
        try:
            with open(temp_path, 'ab' if downloaded > 0 else 'wb') as file:
                with tqdm(total=total_size, initial=downloaded, unit='iB', unit_scale=True, desc=file_name) as pbar:
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
