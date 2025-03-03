from .common import ShowAnythingMie, SaveAnythingAsFile, CompareFiles
from .caption_file_operator import BatchRenameFiles, BatchDeleteFiles, BatchEditTextFiles, BatchSyncImageCaptionFiles, \
    SummaryTextFiles, BatchConvertImageFiles, DedupImageFiles
from .downloader import ModelDownloader
from .utils import add_suffix, add_emoji

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    add_suffix("BatchRenameFiles"): BatchRenameFiles,
    add_suffix("BatchDeleteFiles"): BatchDeleteFiles,
    add_suffix("BatchEditTextFiles"): BatchEditTextFiles,
    add_suffix("BatchSyncImageCaptionFiles"): BatchSyncImageCaptionFiles,
    add_suffix("SummaryTextFiles"): SummaryTextFiles,
    add_suffix("BatchConvertImageFiles"): BatchConvertImageFiles,
    add_suffix("DedupImageFiles"): DedupImageFiles,
    add_suffix("ShowAnything"): ShowAnythingMie,
    add_suffix("SaveAnythingAsFile"): SaveAnythingAsFile,
    add_suffix("CompareFiles"): CompareFiles,
    add_suffix("ModelDownloader"): ModelDownloader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    add_suffix("BatchRenameFiles"): add_emoji("Batch Rename Files"),
    add_suffix("BatchDeleteFiles"): add_emoji("Batch Delete Files"),
    add_suffix("BatchEditTextFiles"): add_emoji("Batch Edit Text Files"),
    add_suffix("BatchSyncImageCaptionFiles"): add_emoji("Batch Sync Image Caption Files"),
    add_suffix("SummaryTextFiles"): add_emoji("Summary Text Files"),
    add_suffix("BatchConvertImageFiles"): add_emoji("Batch Convert Image Files"),
    add_suffix("DedupImageFiles"): add_emoji("Dedup Image Files"),
    add_suffix("ShowAnything"): add_emoji("Show Anything"),
    add_suffix("SaveAnythingAsFile"): add_emoji("Save Anything As File"),
    add_suffix("CompareFiles"): add_emoji("Compare Files"),
    add_suffix("ModelDownloader"): add_emoji("Model Downloader")
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
