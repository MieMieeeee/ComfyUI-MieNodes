from mie_llm.utils import (
    compute_hash,
    convert_size,
    is_image_file,
    load_plugin_config,
    mie_log,
    resolve_token,
)
from mie_llm.media import (
    build_multimodal_user_content,
    image_tensor_batch_to_data_urls,
    image_tensor_to_data_url,
)

LOGO_SUFFIX = "|Mie"
LOGO_EMOJI = "🐑"


def add_suffix(source):
    return source + LOGO_SUFFIX


def add_emoji(source):
    return source + " " + LOGO_EMOJI


# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")


def compute_hash(file_path, hash_algorithm):
    if hash_algorithm == "None":
        return None
    hash_func = getattr(hashlib, hash_algorithm)()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def is_image_file(file_path):
    """
    Check if a file is a valid image using Pillow.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - bool: True if the file is a valid image, False otherwise.
    """
    try:
        with Image.open(file_path) as img:
            return img.format is not None  # Returns True if the image format is valid
    except (IOError, FileNotFoundError):
        return False


def load_plugin_config(filename="mie_llm_keys.json"):
    p = Path(__file__).resolve().parents[1] / filename
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_token(api_token, default_key=None, config_file="mie_llm_keys.json", config_key=None, prefer_local=True):
    cfg = load_plugin_config(config_file or "mie_llm_keys.json")
    k = config_key or default_key
    cfg_token = (cfg.get(k) or "")
    api_token = (api_token or "")
    if prefer_local:
        return (cfg_token or api_token)
    return (api_token or cfg_token)
