from .common import ShowAnythingMie
from .caption_file_operator import BatchRenameFiles, BatchEditTextFiles, BatchSyncImageCaptionFiles
from .utils import add_suffix, add_emoji

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    add_suffix("BatchRenameFiles"): BatchRenameFiles,
    add_suffix("BatchEditTextFiles"): BatchEditTextFiles,
    add_suffix("BatchSyncImageCaptionFiles"): BatchSyncImageCaptionFiles,
    add_suffix("ShowAnything"): ShowAnythingMie
}

NODE_DISPLAY_NAME_MAPPINGS = {
    add_suffix("BatchRenameFiles"): add_emoji("Batch Rename Files"),
    add_suffix("BatchEditTextFiles"): add_emoji("Batch Edit Text Files"),
    add_suffix("BatchSyncImageCaptionFiles"): add_emoji("Batch Sync Image Caption Files"),
    add_suffix("ShowAnything"): add_emoji("Show Anything"),
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
