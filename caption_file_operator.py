import os
import re
import imghdr
from glob import glob

from .utils import mie_log

MY_CATEGORY = "🐑 MieNodes/🐑 Caption Tools"


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
                "prefix": ("STRING",),
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
            the_log_message = "No {} files found in directory {}.".format(file_extension, directory)
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

        the_log_message = "{} files updated.".format(updated_count)
        mie_log(the_log_message)
        return updated_count, the_log_message


class BatchDeleteFiles(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/files"}),
                "file_extension": ("STRING", {"default": ".txt"}),
            },
            "optional": {
                "prefix": ("STRING",),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("deleted_file_count", "log")
    FUNCTION = "batch_delete_files"

    CATEGORY = MY_CATEGORY

    def batch_delete_files(self, directory, file_extension, prefix=None):
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
            if prefix is None or file_name.startswith(prefix):
                os.remove(file_path)
                deleted_count += 1

        the_log_message = f"{deleted_count} files deleted from {directory}."
        mie_log(the_log_message)
        return deleted_count, the_log_message


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
                "target_text": ("STRING",),
                "new_text": ("STRING",),
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
            return 0, f"No {file_extension} files found in {directory}."

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
                    return 0, "Target text is required for Replace operation."
                content = re.sub(target_text, new_text, content)
            elif operation == 'remove':
                if not target_text:
                    return 0, "Target text is required for Remove operation."
                content = content.replace(target_text, '')
            else:
                return 0, f"Unsupported operation: {operation}"

            if content != original_content:
                modified_count += 1
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)

        the_log_message = (f"{operation.capitalize()} operation completed successfully for {modified_count} files "
                           f"in {directory}.")
        mie_log(the_log_message)
        return modified_count, the_log_message


class BatchSyncImageCaptionFiles(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/files"},),
                "caption_content": ("STRING",),
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
            mie_log("{} is {}".format(file_path, imghdr.what(file_path)))
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

        the_log_message = f"Created {created_count} and deleted {deleted_count} captions for files in {directory}."
        mie_log(the_log_message)
        return the_log_message,


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
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "summary_txt_files"

    CATEGORY = MY_CATEGORY

    def summary_txt_files(self, directory, add_separator, save_to_file, file_extension, summary_file_name):
        """
        Summarize text files in a directory.

        Parameters:
        - directory (str): Directory path
        - add_separator (bool): Whether to add a separator between file contents
        - file_extension (str): File extension to operate on (e.g., ".txt")
        - save_to_file (bool): Whether to save the summary to a file
        - summary_file_name (str): Name of the summary file

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
            the_log_message = f"Summarized {len(files)} files in {directory} and saved in {summary_file_path}."
            mie_log(the_log_message)
            return the_log_message,

        the_log_message = f"Summarized {len(files)} files in {directory}."
        mie_log(the_log_message)
        return summary_text,
