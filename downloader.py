import os
import re
import shutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import math
import urllib
from tqdm import tqdm

import folder_paths

from .utils import mie_log
 

MY_CATEGORY = "🐑 MieNodes/🐑 Downloader"


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
        """
        改动要点：
        - 使用 response.raw.read() 小块读取；并在 watcher 中尝试关闭底层 socket，确保能及时打断。
        - 尝试关闭更多底层对象（response.raw._fp.fp.raw 等），以解除阻塞。
        """
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
            comfy_progress = None
            try:
                from comfy import utils as comfy_utils
                steps_total = max(1, math.ceil((total_size - downloaded) / chunk_size)) if total_size > 0 else 0
                comfy_progress = comfy_utils.ProgressBar(steps_total or 1)
            except Exception:
                pass
            with open(temp_path, 'ab' if downloaded > 0 else 'wb') as file:
                tqdm_total = total_size if total_size > 0 else None
                with tqdm(total=tqdm_total, initial=downloaded, unit='iB', unit_scale=True, desc=file_name) as pbar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        size = file.write(data)
                        downloaded += size
                        pbar.update(size)
                        if comfy_progress is not None:
                            try:
                                comfy_progress.update(1)
                            except Exception:
                                pass

            # Verify downloaded size
            if total_size > 0 and downloaded != total_size:
                raise requests.exceptions.ConnectionError(f"Incomplete download: expected {total_size} bytes, got {downloaded} bytes")

            shutil.move(temp_path, full_path)
            return mie_log(f"File {full_path} is downloaded from {url}"),

        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
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


class HFRepoDownloader(object):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url_or_repo_id": ("STRING", {"default": "google/gemma-3-12b-it-qat-q4_0-unquantized"}),
                "save_path": ("STRING", {"default": "checkpoints/gemma-3-12b"}),
                "use_hf_mirror": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "hf_token": ("STRING", {"default": "", "multiline": False, "password": True}),
                "revision": ("STRING", {"default": "main"}),
                "allow_patterns": ("STRING", {"default": "", "multiline": False}),
                "exclude_patterns": ("STRING", {"default": "", "multiline": False}),
                "trigger_signal": (("*", {})),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "download_repo"

    CATEGORY = MY_CATEGORY

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    def download_repo(self, url_or_repo_id, save_path, use_hf_mirror, hf_token, revision, allow_patterns, exclude_patterns, trigger_signal=None):
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return mie_log("Error: huggingface_hub is not installed. Please install it using 'pip install huggingface_hub'."),

        # Extract repo_id if it's a URL
        repo_id = url_or_repo_id.strip()
        if repo_id.startswith("http"):
            # Try to extract repo_id from URL like https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
            # or https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main
            match = re.search(r"huggingface\.co/([^/]+/[^/]+)", repo_id)
            if match:
                repo_id = match.group(1)
            else:
                return mie_log(f"Error: Could not extract repo_id from URL: {url_or_repo_id}"),

        # Setup save directory
        target_directory = os.path.join(folder_paths.models_dir, save_path)
        if not os.path.exists(target_directory):
            os.makedirs(target_directory, exist_ok=True)

        # Setup environment variables for mirror
        env_vars = os.environ.copy()
        if use_hf_mirror:
            env_vars["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        # Prepare parameters
        kwargs = {
            "repo_id": repo_id,
            "local_dir": target_directory,
            "revision": revision if revision else None,
            "token": hf_token if hf_token else None,
            "local_dir_use_symlinks": False,  # Always download actual files
        }

        if allow_patterns:
            kwargs["allow_patterns"] = [p.strip() for p in allow_patterns.split(",") if p.strip()]
        if exclude_patterns:
            kwargs["ignore_patterns"] = [p.strip() for p in exclude_patterns.split(",") if p.strip()]

        try:
            # Temporarily set environment variable if needed
            original_hf_endpoint = os.environ.get("HF_ENDPOINT")
            if use_hf_mirror:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

            downloaded_path = snapshot_download(**kwargs)
            
            # Restore environment variable
            if use_hf_mirror:
                if original_hf_endpoint:
                    os.environ["HF_ENDPOINT"] = original_hf_endpoint
                else:
                    del os.environ["HF_ENDPOINT"]

            return mie_log(f"Successfully downloaded repository {repo_id} to {downloaded_path}"),
        
        except Exception as e:
            # Restore environment variable in case of error
            if use_hf_mirror:
                if original_hf_endpoint:
                    os.environ["HF_ENDPOINT"] = original_hf_endpoint
                else:
                    del os.environ["HF_ENDPOINT"]
            return mie_log(f"Error downloading repository {repo_id}: {str(e)}"),

 
