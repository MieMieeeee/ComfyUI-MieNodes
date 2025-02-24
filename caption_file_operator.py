import os
import re
import imghdr
import shutil
import imagehash
from datetime import datetime
from glob import glob
from PIL import Image

from .utils import mie_log

MY_CATEGORY = "🐑 MieNodes/🐑 Caption Tools"


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class BatchRenameFiles(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/files"}),
                "file_extension": ("STRING", {"default": ".jpg"}),
                "numbering_format": ("STRING", {"default": "####"}),
                "update_caption_as_well": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "prefix": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("updated_file_count", "log")
    FUNCTION = "batch_rename_files"

    CATEGORY = MY_CATEGORY

    def batch_rename_files(self, directory, file_extension, numbering_format, update_caption_as_well, prefix):
        """
        Batch rename files and add prefix and numbering.

        Parameters:
        - directory (str): Directory path
        - file_extension (str): File extension to operate on (e.g., ".jpg", ".txt")
        - numbering_format (str): Numbering format, '###' means three digits
        - update_caption_as_well (bool): Rename the txt file with same name as well
        - prefix (str): File name prefix

        Returns:
        - Number of files updated
        - Log
        """

        updated_count = 0

        files = glob(os.path.join(directory, f"*{file_extension}"))
        files.sort()  # Ensure files are sorted alphabetically

        if not files:
            current_time = get_current_time()
            the_log_message = "No {} files found in directory {} at {}.".format(file_extension, directory, current_time)
            mie_log(the_log_message)
            return 0, the_log_message

        num_digits = numbering_format.count('#')

        for index, file_path in enumerate(files, start=1):
            directory, old_name = os.path.split(file_path)
            new_name = f"{prefix}{str(index).zfill(num_digits)}{file_extension}"
            new_path = os.path.join(directory, new_name)

            os.rename(file_path, new_path)
            updated_count += 1

            if file_extension != ".txt" and update_caption_as_well:
                old_caption_path = os.path.splitext(file_path)[0] + ".txt"
                new_caption_path = os.path.splitext(new_path)[0] + ".txt"
                if os.path.exists(old_caption_path):
                    os.rename(old_caption_path, new_caption_path)

        current_time = get_current_time()
        the_log_message = "{} files updated at {}.".format(updated_count, current_time)
        mie_log(the_log_message)
        return updated_count, the_log_message

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


class BatchDeleteFiles(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/files"}),
                "file_extension": ("STRING", {"default": ".txt"}),
            },
            "optional": {
                "prefix": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("deleted_file_count", "log")
    FUNCTION = "batch_delete_files"

    CATEGORY = MY_CATEGORY

    def batch_delete_files(self, directory, file_extension, prefix):
        """
        Batch delete files with the specified extension and optional prefix.

        Parameters:
        - directory (str): Directory path
        - file_extension (str): File extension to delete (e.g., ".jpg", ".txt")
        - prefix (str, optional): File name prefix to check

        Returns:
        - Number of files deleted
        - Log message
        """

        deleted_count = 0
        files = glob(os.path.join(directory, f"*{file_extension}"))

        for file_path in files:
            file_name = os.path.basename(file_path)
            if len(prefix) > 0 or file_name.startswith(prefix):
                os.remove(file_path)
                deleted_count += 1

        current_time = get_current_time()
        the_log_message = f"{deleted_count} files deleted from {directory} at {current_time}."
        return deleted_count, mie_log(the_log_message)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


class BatchEditTextFiles(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/files"},),
                "operation": (["insert", "append", "replace", "remove"], {}),
            },
            "optional": {
                "file_extension": ("STRING", {"default": ".txt"},),
                "target_text": ("STRING", {"default": ""}),
                "new_text": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("updated_file_count", "log")
    FUNCTION = "edit_text_file"

    CATEGORY = MY_CATEGORY

    def edit_text_file(self, directory, operation, file_extension, target_text, new_text):
        """
        Operate on text files (Insert, Append, Replace, or Remove)

        Parameters:
        - directory (str): Directory path
        - file_extension (str): File extension to operate on (e.g., ".txt")
        - operation (str): Operation type ('insert', 'append', 'replace', 'remove')
        - target_text (str): Target text to replace or remove (only for Replace and Remove operations)
        - new_text (str): New content to insert/append/replace

        Returns:
        - Number of files updated
        - Log
        """

        files = glob(os.path.join(directory, f"*{file_extension}"))

        if not files:
            return 0, mie_log(f"No {file_extension} files found in {directory}.")

        modified_count = 0

        mie_log("Total {} files with {}".format(len(files), file_extension))
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            original_content = content

            if operation == 'insert':
                content = new_text + content
            elif operation == 'append':
                content += new_text
            elif operation == 'replace':
                if not target_text:
                    return 0, mie_log("Target text is required for Replace operation.")
                content = re.sub(target_text, new_text, content)
            elif operation == 'remove':
                if not target_text:
                    return 0, mie_log("Target text is required for Remove operation.")
                content = content.replace(target_text, '')
            else:
                return 0, f"Unsupported operation: {operation}"

            if content != original_content:
                modified_count += 1
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)

        current_time = get_current_time()
        the_log_message = (f"{operation.capitalize()} operation completed successfully for {modified_count} files "
                           f"in {directory} at {current_time}.")
        return modified_count, mie_log(the_log_message)

    # @classmethod
    # def IS_CHANGED(cls, **kwargs):
    #     return float("nan")


class BatchSyncImageCaptionFiles(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/files"},),
                "caption_content": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "sync_image_caption_files"

    CATEGORY = MY_CATEGORY

    def sync_image_caption_files(self, directory, caption_content):
        """
        Synchronize image files and caption files:
        - Generate a matching .txt file for each supported image file (if it doesn't exist)
        - Delete orphaned .txt files that don't have a corresponding image file

        Parameters:
        - directory (str): Directory path
        - caption_content (str): Caption file data, such as "nazha,"

        Returns:
        - Log
        """

        images = set()
        caption_ext = ".txt"
        for file_path in glob(os.path.join(directory, "*")):
            if imghdr.what(file_path):
                images.add(file_path)

        caption_files = set(glob(os.path.join(directory, caption_ext)))

        # Generate matching txt files and write caption
        created_count = 0
        for image_file in images:
            caption_file = os.path.splitext(image_file)[0] + caption_ext
            if caption_file not in caption_files:
                with open(caption_file, 'w', encoding='utf-8') as file:
                    file.write(caption_content)
                created_count += 1

        # Delete orphaned txt files
        deleted_count = 0
        for caption_file in caption_files:
            if not os.path.splitext(caption_file)[0] in {os.path.splitext(img)[0] for img in images}:
                os.remove(caption_file)
                deleted_count += 1

        current_time = get_current_time()
        the_log_message = f"Created {created_count} and deleted {deleted_count} captions for files in {directory} at {current_time}."
        return mie_log(the_log_message),

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


class SummaryTextFiles(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/files"},),
                "add_separator": ("BOOLEAN", {"default": True}),
                "save_to_file": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "file_extension": ("STRING", {"default": ".txt"},),
                "summary_file_name": ("STRING", {"default": "summary.txt"}),
                "trigger_signal": (("*", {})),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "summary_txt_files"

    CATEGORY = MY_CATEGORY

    @classmethod
    def VALIDATE_INPUTS(s, input_types):
        return True

    def summary_txt_files(self, directory, add_separator, save_to_file, file_extension, summary_file_name, trigger_signal):
        """
        Summarize text files in a directory.

        Parameters:
        - directory (str): Directory path
        - add_separator (bool): Whether to add a separator between file contents
        - file_extension (str): File extension to operate on (e.g., ".txt")
        - save_to_file (bool): Whether to save the summary to a file
        - summary_file_name (str): Name of the summary file
        - trigger_signal: Just a trigger

        Returns:
        - Log message
        """

        files = glob(os.path.join(directory, f"*{file_extension}"))

        if not files:
            return f"No {file_extension} files found in {directory}.",

        summary_content = []
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                file_name = os.path.basename(file_path)
                if add_separator:
                    separator = f"=== FILE: {file_name} ===\n"
                    summary_content.append(separator + content)
                else:
                    summary_content.append(content)

        summary_text = "\n".join(summary_content)

        if save_to_file:
            summary_file_path = os.path.join(directory, summary_file_name)
            with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
                summary_file.write(summary_text)
            current_time = get_current_time()
            the_log_message = f"Summarized {len(files)} files in {directory} and saved in {summary_file_path} at {current_time}."
            return mie_log(the_log_message),

        current_time = get_current_time()
        the_log_message = f"Summarized {len(files)} files in {directory} at {current_time}."
        mie_log(the_log_message)
        return summary_text,

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


class BatchConvertImageFiles(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/files"}),
                "target_format": (["jpg", "png"], {"default": "jpg"}),
                "save_original": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("converted_file_count", "log")
    FUNCTION = "convert_image_files"

    CATEGORY = MY_CATEGORY

    def convert_image_files(self, directory, target_format, save_original):
        """
        Convert all images in the specified directory to the target format.

        Parameters:
        - directory (str): Directory path
        - target_format (str): Target image format ('jpg' or 'png')
        - save_original (bool): Whether to save the original files

        Returns:
        - Number of files converted
        - Log message
        """

        supported_formats = ["jpeg", "png", "bmp", "gif", "tiff", "webp"]
        files = [f for f in glob(os.path.join(directory, "*")) if self.is_supported_image(f, supported_formats)]

        if not files:
            return 0, mie_log(f"No supported image files found in {directory}.")

        if save_original:
            backup_dir = os.path.join(directory, "backup")
            os.makedirs(backup_dir, exist_ok=True)

        converted_count = 0
        for file_path in files:
            with Image.open(file_path) as img:
                base_name, ext = os.path.splitext(file_path)
                ext = ext.lower().lstrip('.')
                if ext == target_format:
                    continue  # Skip conversion if the image is already in the target format

                base_name = os.path.splitext(file_path)[0]
                new_file_path = f"{base_name}.{target_format}"
                img.convert("RGB").save(new_file_path, "JPEG" if target_format == "jpg" else target_format.upper())
                converted_count += 1

            if save_original:
                shutil.move(file_path, os.path.join(backup_dir, os.path.basename(file_path)))
            else:
                os.remove(file_path)

        current_time = get_current_time()
        the_log_message = f"Converted {converted_count} images to {target_format} format in {directory} at {current_time}."
        if save_original:
            the_log_message += f" source images are saved in {backup_dir}."
        return converted_count, mie_log(the_log_message)

    @staticmethod
    def is_supported_image(file_path, supported_formats):
        try:
            with Image.open(file_path) as img:
                return img.format.lower() in supported_formats
        except IOError:
            return False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


class DedupImageFiles(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/files"}),
                "max_distance_threshold": ("INT", {"default": 10, "min": 0, "max": 64}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("deleted_file_count", "log")
    FUNCTION = "dedup_image_files"

    CATEGORY = MY_CATEGORY

    def dedup_image_files(self, directory, max_distance_threshold):
        """
        Delete duplicated image files in the specified directory

        Parameters:
        - directory (str): Directory path

        Returns:
        - Number of files deleted
        - Log message
        """

        hashfunc = imagehash.phash
        hashes = {}

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            try:
                with Image.open(filepath) as img:
                    img_hash = hashfunc(img)
                    hashes[filename] = img_hash
            except Exception as e:
                mie_log(f"Unable to process file {filename}: {e}")

        # Compare image hashes and record duplicate images
        duplicates = {}
        filenames = list(hashes.keys())
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                file1, file2 = filenames[i], filenames[j]
                hash1, hash2 = hashes[file1], hashes[file2]
                if hash1 - hash2 <= max_distance_threshold:  # Hamming distance less than or equal to threshold
                    if file1 not in duplicates:
                        duplicates[file1] = []
                    duplicates[file1].append(os.path.join(directory, file2))

        # Delete duplicate images
        files_to_delete = set()
        for key, duplicate_files in duplicates.items():
            # Keep the main file (key) and add the rest of the duplicate files to the delete list
            files_to_delete.update(duplicate_files)

        # Delete files and log
        deleted_count = 0
        for file in files_to_delete:
            try:
                if os.path.exists(file):  # Ensure the file still exists
                    os.remove(file)
                    mie_log(f"Deleted duplicate image: {file}")
                    deleted_count += 1
                else:
                    mie_log(f"File does not exist, may have already been deleted: {file}")

            except Exception as e:
                mie_log(f"Unable to delete file {file}: {e}")

        # Log
        current_time = get_current_time()
        the_log_message = f"Deleted {deleted_count} duplicate images from {directory} at {current_time}."
        return deleted_count, mie_log(the_log_message)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
