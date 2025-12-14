from .common import ShowAnythingMie, SaveAnythingAsFile, CompareFiles, GetAbsolutePath, GetFileInfo, \
    GetDirectoryFilesInfo, CopyFiles, DeleteFiles
from .caption_file_operator import BatchRenameFiles, BatchDeleteFiles, BatchEditTextFiles, BatchSyncImageCaptionFiles, \
    SummaryTextFiles, BatchConvertImageFiles, DedupImageFiles
from .downloader import ModelDownloader
from .translator import TextTranslator
from .prompt_generator import PromptGenerator, KontextPromptGenerator, AddUserKontextPreset, RemoveUserKontextPreset, \
    FrameTransitionPromptGenerator, HunyuanVideoI2VPromptGenerator, HunyuanVideoT2VPromptGenerator, ZImagePromptGenerator, Flux2PromptGenerator
from .llm_service_connector import SetGeneralLLMServiceConnector, SetSiliconFlowLLMServiceConnector, \
    SetGithubModelsLLMServiceConnector, SetZhiPuLLMServiceConnector, SetKimiLLMServiceConnector, \
    SetDeepSeekLLMServiceConnector, SetGeminiLLMServiceConnector, SetBailianLLMServiceConnector, \
    CheckLLMServiceConnectivity, CallLLMService
from .audio_operator import WavConcat
from .string_operator import StringConcat
from .image_operator import SingleImageToVideo, AddNumberWatermarkForImage
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
    add_suffix("ModelDownloader"): ModelDownloader,
    add_suffix("SetGeneralLLMServiceConnector"): SetGeneralLLMServiceConnector,
    add_suffix("SetSiliconFlowLLMServiceConnector"): SetSiliconFlowLLMServiceConnector,
    add_suffix("SetGithubModelsLLMServiceConnector"): SetGithubModelsLLMServiceConnector,
    add_suffix("SetKimiLLMServiceConnector"): SetKimiLLMServiceConnector,
    add_suffix("SetZhiPuLLMServiceConnector"): SetZhiPuLLMServiceConnector,
    add_suffix("SetDeepSeekLLMServiceConnector"): SetDeepSeekLLMServiceConnector,
    add_suffix("SetGeminiLLMServiceConnector"): SetGeminiLLMServiceConnector,
    add_suffix("SetBailianLLMServiceConnector"): SetBailianLLMServiceConnector,
    add_suffix("CheckLLMServiceConnectivity"): CheckLLMServiceConnectivity,
    add_suffix("CallLLMService"): CallLLMService,
    add_suffix("Translator"): TextTranslator,
    add_suffix("PromptGenerator"): PromptGenerator,
    add_suffix("KontextPromptGenerator"): KontextPromptGenerator,
    add_suffix("AddUserKontextPreset"): AddUserKontextPreset,
    add_suffix("RemoveUserKontextPreset"): RemoveUserKontextPreset,
    add_suffix("FrameTransitionPromptGenerator"): FrameTransitionPromptGenerator,
    add_suffix("HunyuanVideoI2VPromptGenerator"): HunyuanVideoI2VPromptGenerator,
    add_suffix("HunyuanVideoT2VPromptGenerator"): HunyuanVideoT2VPromptGenerator,
    add_suffix("ZImagePromptGenerator"): ZImagePromptGenerator,
    add_suffix("Flux2PromptGenerator"): Flux2PromptGenerator,
    add_suffix("GetAbsolutePath"): GetAbsolutePath,
    add_suffix("GetFileInfo"): GetFileInfo,
    add_suffix("GetDirectoryFilesInfo"): GetDirectoryFilesInfo,
    add_suffix("CopyFiles"): CopyFiles,
    add_suffix("DeleteFiles"): DeleteFiles,
    add_suffix("StringConcat"): StringConcat,
    add_suffix("WavConcat"): WavConcat,
    add_suffix("SingleImageToVideo"): SingleImageToVideo,
    add_suffix("AddNumberWatermarkForImage"): AddNumberWatermarkForImage,
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
    add_suffix("SetGeneralLLMServiceConnector"): add_emoji("Set General LLM Service Connector"),
    add_suffix("SetSiliconFlowLLMServiceConnector"): add_emoji("Set SiliconFlow LLM Service Connector"),
    add_suffix("SetGithubModelsLLMServiceConnector"): add_emoji("Set Github Models LLM Service Connector"),
    add_suffix("SetZhiPuLLMServiceConnector"): add_emoji("Set ZhiPu LLM Service Connector"),
    add_suffix("SetKimiLLMServiceConnector"): add_emoji("Set Kimi LLM Service Connector"),
    add_suffix("SetDeepSeekLLMServiceConnector"): add_emoji("Set DeepSeek LLM Service Connector"),
    add_suffix("SetGeminiLLMServiceConnector"): add_emoji("Set Gemini LLM Service Connector"),
    add_suffix("SetBailianLLMServiceConnector"): add_emoji("Set Bailian LLM Service Connector"),
    add_suffix("CheckLLMServiceConnectivity"): add_emoji("Check LLM Service Connectivity"),
    add_suffix("CallLLMService"): add_emoji("Call LLM Service"),
    add_suffix("ModelDownloader"): add_emoji("Model Downloader"),
    add_suffix("Translator"): add_emoji("Translator"),
    add_suffix("PromptGenerator"): add_emoji("Prompt Generator"),
    add_suffix("KontextPromptGenerator"): add_emoji("Kontext Prompt Generator"),
    add_suffix("FrameTransitionPromptGenerator"): add_emoji("Frame Transition Prompt Generator"),
    add_suffix("AddUserKontextPreset"): add_emoji("Add User Kontext Preset"),
    add_suffix("RemoveUserKontextPreset"): add_emoji("Remove User Kontext Preset"),
    add_suffix("HunyuanVideoI2VPromptGenerator"): add_emoji("Hunyuan Video I2V Prompt Generator"),
    add_suffix("HunyuanVideoT2VPromptGenerator"): add_emoji("Hunyuan Video T2V Prompt Generator"),
    add_suffix("ZImagePromptGenerator"): add_emoji("Z-Image Prompt Generator"),
    add_suffix("Flux2PromptGenerator"): add_emoji("Flux2 Prompt Generator"),
    add_suffix("GetAbsolutePath"): add_emoji("Get Absolute Path"),
    add_suffix("GetFileInfo"): add_emoji("Get File Info"),
    add_suffix("GetDirectoryFilesInfo"): add_emoji("Get Directory Files Info"),
    add_suffix("CopyFiles"): add_emoji("Copy Files"),
    add_suffix("DeleteFiles"): add_emoji("Delete Files"),
    add_suffix("StringConcat"): add_emoji("String Concat"),
    add_suffix("WavConcat"): add_emoji("Wav Concat"),
    add_suffix("SingleImageToVideo"): add_emoji("Single Image To Video"),
    add_suffix("AddNumberWatermarkForImage"): add_emoji("Add Number Watermark For Image"),
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
