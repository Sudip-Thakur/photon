"""
File Handler Utilities for Colorization App
Handles file I/O operations, path management, and file validation
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

# Set up logging
logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file operations for the colorization application"""

    # Supported image formats
    SUPPORTED_IMAGE_FORMATS = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".bmp": "BMP",
        ".tiff": "TIFF",
        ".tif": "TIFF",
        ".webp": "WEBP",
    }

    # Supported model formats
    SUPPORTED_MODEL_FORMATS = {
        ".pth": "PyTorch",
        ".pt": "PyTorch",
        ".pkl": "Pickle",
        ".ckpt": "Checkpoint",
    }

    def __init__(self, base_directory: Optional[str] = None):
        """
        Initialize file handler

        Args:
            base_directory: Base directory for operations (defaults to home)
        """
        self.base_directory = Path(base_directory) if base_directory else Path.home()
        self.ensure_directory_exists(self.base_directory)

    def ensure_directory_exists(self, directory: Union[str, Path]) -> bool:
        """
        Ensure directory exists, create if it doesn't

        Args:
            directory: Directory path

        Returns:
            True if directory exists/was created, False otherwise
        """
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            return False

    def is_valid_image_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file is a valid image format

        Args:
            file_path: Path to file

        Returns:
            True if valid image format, False otherwise
        """
        return Path(file_path).suffix.lower() in self.SUPPORTED_IMAGE_FORMATS

    def is_valid_model_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file is a valid model format

        Args:
            file_path: Path to file

        Returns:
            True if valid model format, False otherwise
        """
        return Path(file_path).suffix.lower() in self.SUPPORTED_MODEL_FORMATS

    def get_image_files_in_directory(
        self, directory: Union[str, Path], recursive: bool = False
    ) -> List[Path]:
        """
        Get all image files in a directory

        Args:
            directory: Directory to search
            recursive: Whether to search subdirectories

        Returns:
            List of image file paths
        """
        directory = Path(directory)
        image_files = []

        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Directory does not exist: {directory}")
            return image_files

        try:
            if recursive:
                for ext in self.SUPPORTED_IMAGE_FORMATS:
                    image_files.extend(directory.rglob(f"*{ext}"))
                    image_files.extend(directory.rglob(f"*{ext.upper()}"))
            else:
                for ext in self.SUPPORTED_IMAGE_FORMATS:
                    image_files.extend(directory.glob(f"*{ext}"))
                    image_files.extend(directory.glob(f"*{ext.upper()}"))

            # Remove duplicates and sort
            image_files = sorted(list(set(image_files)))
            logger.info(f"Found {len(image_files)} image files in {directory}")

        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")

        return image_files

    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get detailed information about a file

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)
        info = {
            "name": file_path.name,
            "stem": file_path.stem,
            "suffix": file_path.suffix,
            "parent": str(file_path.parent),
            "exists": file_path.exists(),
            "is_file": file_path.is_file(),
            "is_dir": file_path.is_dir(),
        }

        if file_path.exists():
            try:
                stat = file_path.stat()
                info.update(
                    {
                        "size": stat.st_size,
                        "size_mb": stat.st_size / (1024 * 1024),
                        "modified": datetime.fromtimestamp(stat.st_mtime),
                        "created": datetime.fromtimestamp(stat.st_ctime),
                        "is_readable": os.access(file_path, os.R_OK),
                        "is_writable": os.access(file_path, os.W_OK),
                    }
                )
            except Exception as e:
                logger.error(f"Error getting file stats for {file_path}: {e}")

        return info

    def generate_unique_filename(
        self, directory: Union[str, Path], base_name: str, extension: str
    ) -> Path:
        """
        Generate a unique filename in the given directory

        Args:
            directory: Target directory
            base_name: Base filename (without extension)
            extension: File extension (with or without dot)

        Returns:
            Unique file path
        """
        directory = Path(directory)

        # Ensure extension starts with dot
        if not extension.startswith("."):
            extension = f".{extension}"

        # Start with base name
        filename = f"{base_name}{extension}"
        file_path = directory / filename

        # If file doesn't exist, return it
        if not file_path.exists():
            return file_path

        # Generate unique name with counter
        counter = 1
        while file_path.exists():
            filename = f"{base_name}_{counter:03d}{extension}"
            file_path = directory / filename
            counter += 1

        return file_path

    def copy_file(
        self,
        source: Union[str, Path],
        destination: Union[str, Path],
        overwrite: bool = False,
    ) -> bool:
        """
        Copy file from source to destination

        Args:
            source: Source file path
            destination: Destination file path
            overwrite: Whether to overwrite existing files

        Returns:
            True if successful, False otherwise
        """
        try:
            source = Path(source)
            destination = Path(destination)

            if not source.exists():
                logger.error(f"Source file does not exist: {source}")
                return False

            # Create destination directory if needed
            self.ensure_directory_exists(destination.parent)

            # Check if destination exists
            if destination.exists() and not overwrite:
                logger.warning(
                    f"Destination file exists and overwrite=False: {destination}"
                )
                return False

            # Copy file
            shutil.copy2(source, destination)
            logger.info(f"File copied: {source} -> {destination}")
            return True

        except Exception as e:
            logger.error(f"Error copying file {source} -> {destination}: {e}")
            return False

    def move_file(
        self,
        source: Union[str, Path],
        destination: Union[str, Path],
        overwrite: bool = False,
    ) -> bool:
        """
        Move file from source to destination

        Args:
            source: Source file path
            destination: Destination file path
            overwrite: Whether to overwrite existing files

        Returns:
            True if successful, False otherwise
        """
        try:
            source = Path(source)
            destination = Path(destination)

            if not source.exists():
                logger.error(f"Source file does not exist: {source}")
                return False

            # Create destination directory if needed
            self.ensure_directory_exists(destination.parent)

            # Check if destination exists
            if destination.exists() and not overwrite:
                logger.warning(
                    f"Destination file exists and overwrite=False: {destination}"
                )
                return False

            # Move file
            shutil.move(str(source), str(destination))
            logger.info(f"File moved: {source} -> {destination}")
            return True

        except Exception as e:
            logger.error(f"Error moving file {source} -> {destination}: {e}")
            return False

    def delete_file(self, file_path: Union[str, Path], confirm: bool = True) -> bool:
        """
        Delete a file

        Args:
            file_path: Path to file to delete
            confirm: Whether file should exist (False allows deleting non-existent files)

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                if confirm:
                    logger.warning(f"File does not exist: {file_path}")
                    return False
                else:
                    return True  # Consider it successful if file doesn't exist

            file_path.unlink()
            logger.info(f"File deleted: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False

    def save_json(
        self, data: Any, file_path: Union[str, Path], indent: int = 2
    ) -> bool:
        """
        Save data as JSON file

        Args:
            data: Data to save
            file_path: Output file path
            indent: JSON indentation

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            self.ensure_directory_exists(file_path.parent)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

            logger.info(f"JSON saved: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving JSON {file_path}: {e}")
            return False

    def load_json(self, file_path: Union[str, Path]) -> Optional[Any]:
        """
        Load data from JSON file

        Args:
            file_path: JSON file path

        Returns:
            Loaded data or None if failed
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                logger.warning(f"JSON file does not exist: {file_path}")
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"JSON loaded: {file_path}")
            return data

        except Exception as e:
            logger.error(f"Error loading JSON {file_path}: {e}")
            return None

    def save_yaml(self, data: Any, file_path: Union[str, Path]) -> bool:
        """
        Save data as YAML file

        Args:
            data: Data to save
            file_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            self.ensure_directory_exists(file_path.parent)

            with open(file_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    data, f, default_flow_style=False, allow_unicode=True, indent=2
                )

            logger.info(f"YAML saved: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving YAML {file_path}: {e}")
            return False

    def load_yaml(self, file_path: Union[str, Path]) -> Optional[Any]:
        """
        Load data from YAML file

        Args:
            file_path: YAML file path

        Returns:
            Loaded data or None if failed
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                logger.warning(f"YAML file does not exist: {file_path}")
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            logger.info(f"YAML loaded: {file_path}")
            return data

        except Exception as e:
            logger.error(f"Error loading YAML {file_path}: {e}")
            return None

    def save_text(
        self, text: str, file_path: Union[str, Path], encoding: str = "utf-8"
    ) -> bool:
        """
        Save text to file

        Args:
            text: Text content to save
            file_path: Output file path
            encoding: Text encoding

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            self.ensure_directory_exists(file_path.parent)

            with open(file_path, "w", encoding=encoding) as f:
                f.write(text)

            logger.info(f"Text file saved: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving text file {file_path}: {e}")
            return False

    def load_text(
        self, file_path: Union[str, Path], encoding: str = "utf-8"
    ) -> Optional[str]:
        """
        Load text from file

        Args:
            file_path: Text file path
            encoding: Text encoding

        Returns:
            File content or None if failed
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                logger.warning(f"Text file does not exist: {file_path}")
                return None

            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            logger.info(f"Text file loaded: {file_path}")
            return content

        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return None

    def get_directory_size(self, directory: Union[str, Path]) -> Tuple[int, int]:
        """
        Get total size of directory and file count

        Args:
            directory: Directory path

        Returns:
            Tuple of (total_size_bytes, file_count)
        """
        try:
            directory = Path(directory)
            total_size = 0
            file_count = 0

            if directory.exists() and directory.is_dir():
                for file_path in directory.rglob("*"):
                    if file_path.is_file():
                        try:
                            total_size += file_path.stat().st_size
                            file_count += 1
                        except (OSError, IOError):
                            # Skip files that can't be accessed
                            pass

            return total_size, file_count

        except Exception as e:
            logger.error(f"Error calculating directory size {directory}: {e}")
            return 0, 0

    def clean_directory(
        self, directory: Union[str, Path], pattern: str = "*", older_than_days: int = 0
    ) -> Tuple[int, int]:
        """
        Clean directory by removing files matching pattern

        Args:
            directory: Directory to clean
            pattern: File pattern to match
            older_than_days: Only delete files older than this many days

        Returns:
            Tuple of (files_deleted, bytes_freed)
        """
        try:
            directory = Path(directory)
            files_deleted = 0
            bytes_freed = 0

            if not directory.exists() or not directory.is_dir():
                return files_deleted, bytes_freed

            cutoff_time = None
            if older_than_days > 0:
                from datetime import timedelta

                cutoff_time = datetime.now() - timedelta(days=older_than_days)

            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    try:
                        # Check age if specified
                        if cutoff_time:
                            file_time = datetime.fromtimestamp(
                                file_path.stat().st_mtime
                            )
                            if file_time > cutoff_time:
                                continue

                        # Get file size before deletion
                        file_size = file_path.stat().st_size

                        # Delete file
                        file_path.unlink()

                        files_deleted += 1
                        bytes_freed += file_size

                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")

            logger.info(
                f"Cleaned {directory}: {files_deleted} files, {bytes_freed} bytes"
            )
            return files_deleted, bytes_freed

        except Exception as e:
            logger.error(f"Error cleaning directory {directory}: {e}")
            return 0, 0

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human readable format

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted size string
        """
        if size_bytes == 0:
            return "0 B"

        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        size = float(size_bytes)

        while size >= 1024.0 and unit_index < len(units) - 1:
            size /= 1024.0
            unit_index += 1

        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        else:
            return f"{size:.2f} {units[unit_index]}"

    def backup_file(
        self,
        file_path: Union[str, Path],
        backup_directory: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        Create a backup of a file

        Args:
            file_path: File to backup
            backup_directory: Directory for backup (defaults to same directory)

        Returns:
            Backup file path or None if failed
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                logger.error(f"Cannot backup non-existent file: {file_path}")
                return None

            # Determine backup location
            if backup_directory:
                backup_dir = Path(backup_directory)
                self.ensure_directory_exists(backup_dir)
            else:
                backup_dir = file_path.parent

            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
            backup_path = backup_dir / backup_name

            # Copy file
            if self.copy_file(file_path, backup_path):
                logger.info(f"Backup created: {backup_path}")
                return backup_path
            else:
                return None

        except Exception as e:
            logger.error(f"Error creating backup of {file_path}: {e}")
            return None
