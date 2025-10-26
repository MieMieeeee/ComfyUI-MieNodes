MY_CATEGORY = "🐑 MieNodes/🐑 Audio Operator"

try:
    import torch
except ImportError:
    # 确保在 ComfyUI 环境中 torch 可用
    raise ImportError("PyTorch (torch) is required for audio processing nodes but was not found.")


class WavConcat(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO", {"force_output": True}),
            },
            "optional": {
                "audio2": ("AUDIO", {"force_output": True}),
                "mute_audio1": ("BOOLEAN", {"default": False, "label": "Mute Audio 1"}),
                "mute_audio2": ("BOOLEAN", {"default": False, "label": "Mute Audio 2"}),
                # 新增：开始间隔 (start)
                "start_spacer": ("FLOAT",
                                 {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1, "label": "Start Silence (s)"}),
                # 新增：中间间隔 (middle)
                "middle_spacer": ("FLOAT",
                                  {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1, "label": "Middle Silence (s)"}),
                # 新增：结束间隔 (end)
                "end_spacer": ("FLOAT",
                               {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1, "label": "End Silence (s)"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("concatenated_audio",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def _create_silence_tensor(self, duration, sample_rate, reference_tensor):
        """根据时长和参考张量创建静音张量"""
        if duration <= 0.0:
            return None

        num_silent_samples = int(duration * sample_rate)

        # 保持 Batch 和 Channel 维度不变，时间维度设为静音长度
        silent_shape = list(reference_tensor.shape)
        silent_shape[-1] = num_silent_samples

        # 创建全零静音张量，保持 dtype 和 device 一致
        silence_tensor = torch.zeros(tuple(silent_shape),
                                     dtype=reference_tensor.dtype,
                                     device=reference_tensor.device)
        return silence_tensor

    def execute(self, audio1, audio2=None, mute_audio1=False, mute_audio2=False, start_spacer=0.0, middle_spacer=0.0,
                end_spacer=0.0):

        # 1. 提取 audio1 数据
        sample_rate = audio1["sample_rate"]
        waveform1 = audio1["waveform"]

        # 如果 mute_audio1 为 True，则将 waveform1 替换为静音
        if mute_audio1:
            duration1 = waveform1.shape[-1] / sample_rate
            waveform1 = self._create_silence_tensor(duration1, sample_rate, waveform1)

        # 2. 初始化拼接列表
        parts_to_concat = []

        # a. 开始间隔
        start_silence = self._create_silence_tensor(start_spacer, sample_rate, waveform1)
        if start_silence is not None:
            parts_to_concat.append(start_silence)

        # b. Audio 1
        if waveform1 is not None:
            parts_to_concat.append(waveform1)

        # 3. 处理可选的 audio2
        if audio2 is not None:
            # 验证采样率
            if sample_rate != audio2["sample_rate"]:
                raise ValueError(
                    f"Audio sample rate mismatch: audio1 has {sample_rate} Hz, but audio2 has {audio2['sample_rate']} Hz.")

            waveform2 = audio2["waveform"]

            # 如果 mute_audio2 为 True，则将 waveform2 替换为静音
            if mute_audio2:
                duration2 = waveform2.shape[-1] / sample_rate
                waveform2 = self._create_silence_tensor(duration2, sample_rate, waveform2)

            # c. 中间间隔
            middle_silence = self._create_silence_tensor(middle_spacer, sample_rate, waveform1)
            if middle_silence is not None:
                parts_to_concat.append(middle_silence)

            # d. Audio 2
            if waveform2 is not None:
                parts_to_concat.append(waveform2)

        # e. 结束间隔
        end_silence = self._create_silence_tensor(end_spacer, sample_rate, waveform1)
        if end_silence is not None:
            parts_to_concat.append(end_silence)

        # 4. 执行拼接
        if not parts_to_concat:
            # 如果所有输入都无效，则返回一个空的音频
            return ({"waveform": torch.empty(1, 2, 0), "sample_rate": sample_rate},)

        # 沿着最后一个维度（时间维度）拼接所有部分
        final_waveform = torch.cat(parts_to_concat, dim=-1)

        # 5. 返回新的 AUDIO 结构
        result_audio = {
            "waveform": final_waveform,
            "sample_rate": sample_rate
        }

        return (result_audio,)