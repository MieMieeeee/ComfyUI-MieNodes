import json
import urllib.request
import urllib.error
import os
import tempfile
import torchaudio
import torch
try:
    from .utils import mie_log
except ImportError:
    from utils import mie_log

MY_CATEGORY = "🐑 MieNodes/🐑 TTS Service"

class QwenTTSNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "你好，我是通义千问语音合成助手。"}),
                "voice": (
                    [
                        "Cherry", "Ethan", "Chelsie", "Serena", "Dylan", "Jada", "Sunny",
                        "Momo", "Ono Anna", "Vivian", "Eldric Sage", "Bunny", "Elias",
                        "Nofish", "Jennifer", "Ryan", "Katerina", "Li", "Marcus", 
                        "Roy", "Peter", "Rocky", "Kiki", "Eric"
                    ],
                    {"default": "Cherry"}
                ),
                "model": (["qwen3-tts-flash"], {"default": "qwen3-tts-flash"}),
                "language_type": (["Chinese", "English"], {"default": "Chinese"}),
            },
            "optional": {
                "tts_connector": ("TTSConnector",),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Optional if connector is used"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, text, voice, model, language_type, tts_connector=None, api_key=""):
        # 1. Resolve API Key
        resolved_api_key = ""
        if tts_connector:
            # Try to get token from connector object
            if hasattr(tts_connector, "api_token") and tts_connector.api_token:
                resolved_api_key = tts_connector.api_token
        
        if not resolved_api_key:
            if api_key and api_key.strip():
                resolved_api_key = api_key.strip()
            else:
                # Fallback to env var if available
                resolved_api_key = os.environ.get("DASHSCOPE_API_KEY", "")

        if not resolved_api_key:
            raise ValueError("API Key is missing. Please provide a TTS Connector, an API Key input, or set DASHSCOPE_API_KEY env var.")

        # 2. Prepare Request
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        
        headers = {
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "input": {
                "text": text,
                "voice": voice,
                "language_type": language_type
            }
        }
        
        data = json.dumps(payload).encode("utf-8")
        
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        
        mie_log(f"Sending Qwen TTS request to {url} with model {model} and voice {voice}...")
        
        try:
            with urllib.request.urlopen(req) as response:
                response_body = response.read().decode("utf-8")
                response_json = json.loads(response_body)
                
                # Check for API errors in JSON
                if "code" in response_json and response_json["code"]:
                     # Standard DashScope success usually has "output" and no "code" or "code"=="".
                     pass

                if "output" not in response_json:
                    raise ValueError(f"Unexpected response format: {response_body}")
                
                audio_url = None
                output = response_json.get("output", {})
                
                # Check different response formats
                if "audio_url" in output:
                    audio_url = output["audio_url"]
                elif "audio" in output and "url" in output["audio"]:
                     # Format: output: { audio: { url: "..." } }
                    audio_url = output["audio"]["url"]
                elif "results" in output and len(output["results"]) > 0:
                    # Sometimes results[0]['url']
                    audio_url = output["results"][0].get("url") or output["results"][0].get("audio_url")
                
                if not audio_url:
                     raise ValueError(f"Could not find audio_url in response: {response_body}")
                
                mie_log(f"Got audio URL: {audio_url}")
                
                # 3. Download Audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                    temp_path = temp_file.name
                
                # Download using urllib
                urllib.request.urlretrieve(audio_url, temp_path)
                
                # 4. Load Audio
                try:
                    # Try loading with explicit backend if possible, or default
                    try:
                        waveform, sample_rate = torchaudio.load(temp_path, backend="soundfile")
                    except:
                        waveform, sample_rate = torchaudio.load(temp_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to load audio from {temp_path}: {e}. Ensure torchaudio and ffmpeg are installed.")
                finally:
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                # 5. Return Audio
                # Add batch dimension if needed
                if waveform.ndim == 2:
                    waveform = waveform.unsqueeze(0) # [1, C, N]
                
                return ({"waveform": waveform, "sample_rate": sample_rate},)

        except urllib.error.HTTPError as e:
            error_content = e.read().decode("utf-8")
            raise RuntimeError(f"HTTP Error {e.code}: {error_content}")
        except Exception as e:
            raise RuntimeError(f"Qwen TTS Execution Failed: {str(e)}")
