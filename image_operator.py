import cv2
import numpy as np
import torch
import folder_paths
import os
import re
import subprocess
from pathlib import Path
from .utils import mie_log

MY_CATEGORY = "🐑 MieNodes/🐑 Image Operator"

class SingleImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "video"}),
                "save_output": ("BOOLEAN", {"default": True}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 60}),
                "duration": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10000.0}),
                "translation_x": ("FLOAT", {"default": 0, "min": -100, "max": 100}),
                "translation_y": ("FLOAT", {"default": 0, "min": -100, "max": 100}),
                "scale": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 10.0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "create_video_from_images"
    CATEGORY = MY_CATEGORY
    OUTPUT_NODE = True

    def tensor_to_cv2(self, img_tensor):
        # Convert tensor from [H, W, C] and RGB to [H, W, C] and BGR for OpenCV
        img_np = img_tensor.cpu().numpy() * 255.0
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    def _translate_image(self, img, translation):
        height, width = img.shape[:2]
        tx = int(width * translation[0] / 100)
        ty = int(height * translation[1] / 100)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(img, M, (width, height))
        return translated

    def _crop_black_borders(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]

    def _scale_image(self, img, scale, target_size):
        if scale <= 1:
            return cv2.resize(img, target_size)

        height, width = img.shape[:2]
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)
        scaled = cv2.resize(img, (scaled_width, scaled_height))

        start_x = (scaled_width - target_size[0]) // 2
        start_y = (scaled_height - target_size[1]) // 2
        cropped = scaled[start_y:start_y+target_size[1], start_x:start_x+target_size[0]]
        return cropped

    def _transform_image(self, img, translation, scale):
        original_size = (img.shape[1], img.shape[0])
        translated = self._translate_image(img, translation)
        cropped = self._crop_black_borders(translated)
        transformed = self._scale_image(cropped, scale, original_size)
        return transformed

    def _improve_video_quality(self, video_path):
        temp_path = video_path + '.temp.mp4'
        cmd = [
            'ffmpeg', '-i', video_path,
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
            '-y', temp_path
        ]
        try:
            mie_log(f"Improving video quality for {video_path}")
            subprocess.run(cmd, check=True)
            os.replace(temp_path, video_path)
            mie_log(f"Video quality improved for {video_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            mie_log(f"ffmpeg command failed: {e}. The original video is kept.")
            if os.path.exists(temp_path):
                os.remove(temp_path)


    def create_video_from_images(self, images, filename_prefix, save_output, fps, duration, translation_x, translation_y, scale):
        if not images.any():
            raise ValueError("No images provided to create video.")

        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)

        # comfy counter workaround
        max_counter = 0
        if os.path.exists(full_output_folder):
            matcher = re.compile(f"{re.escape(filename)}_(\d+)\D*\.mp4", re.IGNORECASE)
            for f in os.listdir(full_output_folder):
                match = matcher.match(f)
                if match:
                    max_counter = max(max_counter, int(match.group(1)))

        counter = max_counter + 1
        output_filename = f"{filename}_{counter:05d}.mp4"
        output_path = os.path.join(full_output_folder, output_filename)

        mie_log(f"Start creating video with {len(images)} images.")
        mie_log(f"Parameters: output_path={output_path}, fps={fps}, duration={duration}, translation_x={translation_x}, translation_y={translation_y}, scale={scale}")

        # Ensure output directory exists
        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder)

        first_image_cv2 = self.tensor_to_cv2(images[0])
        reference_size = (first_image_cv2.shape[1], first_image_cv2.shape[0])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, reference_size)

        try:
            for i in range(images.shape[0]):
                img_cv2 = self.tensor_to_cv2(images[i])

                transformed = self._transform_image(
                    img_cv2,
                    [translation_x, translation_y],
                    scale
                )

                if transformed.shape[:2] != (reference_size[1], reference_size[0]):
                    transformed = cv2.resize(transformed, reference_size)

                n_frames = int(duration * fps)
                for _ in range(n_frames):
                    out.write(transformed)
        finally:
            out.release()

        self._improve_video_quality(output_path)

        mie_log(f"Video created successfully at {output_path}")
        return (output_path,)
