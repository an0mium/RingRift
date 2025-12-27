"""Tests for app.storage.backends module."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from app.storage.backends import (
    LocalStorage,
    S3Storage,
    GCSStorage,
    StorageBackend,
    get_storage_backend,
    get_storage_from_uri,
)


class TestStorageBackendInterface:
    """Tests for StorageBackend abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that StorageBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            StorageBackend()

    def test_interface_has_required_methods(self):
        """Test that StorageBackend defines all required abstract methods."""
        required_methods = ["upload", "download", "list", "exists", "delete"]
        for method in required_methods:
            assert hasattr(StorageBackend, method)
            assert callable(getattr(StorageBackend, method))


class TestLocalStorage:
    """Tests for LocalStorage backend."""

    def test_initialization(self, tmp_path: Path):
        """Test LocalStorage initialization."""
        storage = LocalStorage(base_path=tmp_path)
        assert storage._base_path == tmp_path.resolve()

    def test_initialization_creates_directory(self, tmp_path: Path):
        """Test that initialization creates base directory if it doesn't exist."""
        nested_path = tmp_path / "nested" / "dir"
        storage = LocalStorage(base_path=nested_path)
        assert nested_path.exists()
        assert nested_path.is_dir()

    def test_upload_creates_parent_directories(self, tmp_path: Path):
        """Test that upload creates parent directories."""
        storage = LocalStorage(base_path=tmp_path)

        # Create source file
        source_file = tmp_path / "source.txt"
        source_file.write_text("test content")

        # Upload to nested path
        storage.upload(source_file, "nested/dir/dest.txt")

        dest_file = tmp_path / "nested" / "dir" / "dest.txt"
        assert dest_file.exists()
        assert dest_file.read_text() == "test content"

    def test_upload_copies_file(self, tmp_path: Path):
        """Test that upload copies file content correctly."""
        storage = LocalStorage(base_path=tmp_path)

        source_file = tmp_path / "source.txt"
        source_file.write_text("hello world")

        storage.upload(source_file, "dest.txt")

        dest_file = tmp_path / "dest.txt"
        assert dest_file.exists()
        assert dest_file.read_text() == "hello world"

    def test_upload_skips_if_same_path(self, tmp_path: Path):
        """Test that upload skips copy if source and dest are same path."""
        storage = LocalStorage(base_path=tmp_path)

        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        # Upload to same location should not error
        storage.upload(file_path, "file.txt")
        assert file_path.read_text() == "content"

    def test_download_copies_file(self, tmp_path: Path):
        """Test that download copies file correctly."""
        storage = LocalStorage(base_path=tmp_path)

        # Create remote file
        remote_file = tmp_path / "remote.txt"
        remote_file.write_text("remote content")

        # Download to different location
        local_file = tmp_path / "local.txt"
        storage.download("remote.txt", local_file)

        assert local_file.exists()
        assert local_file.read_text() == "remote content"

    def test_download_raises_on_missing_file(self, tmp_path: Path):
        """Test that download raises FileNotFoundError for missing files."""
        storage = LocalStorage(base_path=tmp_path)

        with pytest.raises(FileNotFoundError):
            storage.download("nonexistent.txt", tmp_path / "dest.txt")

    def test_download_creates_parent_directories(self, tmp_path: Path):
        """Test that download creates parent directories."""
        storage = LocalStorage(base_path=tmp_path)

        remote_file = tmp_path / "remote.txt"
        remote_file.write_text("content")

        local_file = tmp_path / "nested" / "local.txt"
        storage.download("remote.txt", local_file)

        assert local_file.exists()

    def test_download_skips_if_same_path(self, tmp_path: Path):
        """Test that download skips copy if source and dest are same."""
        storage = LocalStorage(base_path=tmp_path)

        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        storage.download("file.txt", file_path)
        assert file_path.read_text() == "content"

    def test_list_returns_files_in_directory(self, tmp_path: Path):
        """Test listing files in a directory."""
        storage = LocalStorage(base_path=tmp_path)

        # Create test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.txt").write_text("content3")

        files = storage.list("")
        assert len(files) == 3
        assert "file1.txt" in files
        assert "file2.txt" in files
        assert str(Path("subdir") / "file3.txt") in files

    def test_list_with_prefix(self, tmp_path: Path):
        """Test listing files with prefix filter."""
        storage = LocalStorage(base_path=tmp_path)

        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "model1.pt").write_text("model1")
        (tmp_path / "models" / "model2.pt").write_text("model2")
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "data.npz").write_text("data")

        models = storage.list("models")
        assert len(models) == 2
        assert str(Path("models") / "model1.pt") in models
        assert str(Path("models") / "model2.pt") in models

    def test_list_returns_single_file_if_path_is_file(self, tmp_path: Path):
        """Test that list returns single file if path points to file."""
        storage = LocalStorage(base_path=tmp_path)

        file_path = tmp_path / "single.txt"
        file_path.write_text("content")

        files = storage.list("single.txt")
        assert files == ["single.txt"]

    def test_list_returns_empty_for_nonexistent_prefix(self, tmp_path: Path):
        """Test that list returns empty list for nonexistent prefix."""
        storage = LocalStorage(base_path=tmp_path)

        files = storage.list("nonexistent")
        assert files == []

    def test_exists_returns_true_for_existing_file(self, tmp_path: Path):
        """Test exists returns True for existing files."""
        storage = LocalStorage(base_path=tmp_path)

        file_path = tmp_path / "exists.txt"
        file_path.write_text("content")

        assert storage.exists("exists.txt") is True

    def test_exists_returns_false_for_missing_file(self, tmp_path: Path):
        """Test exists returns False for missing files."""
        storage = LocalStorage(base_path=tmp_path)

        assert storage.exists("missing.txt") is False

    def test_delete_removes_file(self, tmp_path: Path):
        """Test that delete removes file."""
        storage = LocalStorage(base_path=tmp_path)

        file_path = tmp_path / "delete_me.txt"
        file_path.write_text("content")

        assert file_path.exists()
        storage.delete("delete_me.txt")
        assert not file_path.exists()

    def test_delete_does_nothing_for_missing_file(self, tmp_path: Path):
        """Test that delete doesn't raise error for missing files."""
        storage = LocalStorage(base_path=tmp_path)

        # Should not raise
        storage.delete("nonexistent.txt")

    def test_download_if_newer_downloads_when_local_missing(self, tmp_path: Path):
        """Test download_if_newer downloads when local file doesn't exist."""
        storage = LocalStorage(base_path=tmp_path)

        remote_file = tmp_path / "remote.txt"
        remote_file.write_text("content")

        local_file = tmp_path / "local.txt"
        result = storage.download_if_newer("remote.txt", local_file)

        assert result is True
        assert local_file.exists()

    def test_download_if_newer_always_downloads_for_local_storage(self, tmp_path: Path):
        """Test download_if_newer always downloads for LocalStorage (no metadata)."""
        storage = LocalStorage(base_path=tmp_path)

        remote_file = tmp_path / "remote.txt"
        remote_file.write_text("content")

        local_file = tmp_path / "local.txt"
        local_file.write_text("old content")

        # Default implementation always downloads
        result = storage.download_if_newer("remote.txt", local_file)
        assert result is True


class TestS3Storage:
    """Tests for S3Storage backend.

    Note: These tests use mocking since boto3 may not be installed.
    """

    def test_initialization_requires_boto3(self):
        """Test that S3Storage raises ImportError without boto3."""
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="boto3 is required"):
                S3Storage(bucket="test-bucket")


class TestGCSStorage:
    """Tests for GCSStorage backend.

    Note: These tests use mocking since google-cloud-storage may not be installed.
    """

    def test_initialization_requires_gcs_library(self):
        """Test that GCSStorage raises ImportError without google-cloud-storage."""
        with patch.dict("sys.modules", {"google.cloud.storage": None, "google.cloud": None, "google": None}):
            with pytest.raises(ImportError, match="google-cloud-storage is required"):
                GCSStorage(bucket="test-bucket")


class TestGetStorageBackend:
    """Tests for get_storage_backend factory function."""

    def test_returns_local_storage_by_default(self, tmp_path: Path):
        """Test that local storage is returned by default."""
        storage = get_storage_backend(base_path=str(tmp_path))
        assert isinstance(storage, LocalStorage)

    def test_returns_local_storage_explicitly(self, tmp_path: Path):
        """Test returning local storage explicitly."""
        storage = get_storage_backend(backend="local", base_path=str(tmp_path))
        assert isinstance(storage, LocalStorage)

    def test_s3_requires_bucket(self):
        """Test that S3 backend requires bucket parameter."""
        with pytest.raises(ValueError, match="STORAGE_BUCKET is required"):
            get_storage_backend(backend="s3")

    def test_gcs_requires_bucket(self):
        """Test that GCS backend requires bucket parameter."""
        with pytest.raises(ValueError, match="STORAGE_BUCKET is required"):
            get_storage_backend(backend="gcs")

    def test_raises_on_unknown_backend(self):
        """Test that unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown storage backend"):
            get_storage_backend(backend="unknown")

    def test_backend_case_insensitive(self, tmp_path: Path):
        """Test that backend type is case-insensitive."""
        storage = get_storage_backend(backend="LOCAL", base_path=str(tmp_path))
        assert isinstance(storage, LocalStorage)

    def test_uses_environment_variables(self, tmp_path: Path, monkeypatch):
        """Test that environment variables are used as defaults."""
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("STORAGE_BASE_PATH", str(tmp_path))

        storage = get_storage_backend()
        assert isinstance(storage, LocalStorage)

    def test_explicit_params_override_env(self, tmp_path: Path, monkeypatch):
        """Test that explicit parameters override environment variables."""
        monkeypatch.setenv("STORAGE_BACKEND", "s3")

        storage = get_storage_backend(backend="local", base_path=str(tmp_path))
        assert isinstance(storage, LocalStorage)


class TestGetStorageFromUri:
    """Tests for get_storage_from_uri function."""

    def test_parses_local_file_uri(self, tmp_path: Path):
        """Test parsing file:// URI."""
        test_path = tmp_path / "data" / "models"
        storage = get_storage_from_uri(f"file://{test_path}")
        assert isinstance(storage, LocalStorage)
        assert storage._base_path == test_path

    def test_parses_plain_path_as_local(self, tmp_path: Path):
        """Test parsing plain path as local storage."""
        test_path = tmp_path / "models"
        storage = get_storage_from_uri(str(test_path))
        assert isinstance(storage, LocalStorage)

    def test_raises_on_unsupported_scheme(self):
        """Test that unsupported URI scheme raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported storage URI scheme"):
            get_storage_from_uri("ftp://example.com/file")

    def test_handles_empty_scheme_as_local(self):
        """Test that URI without scheme is treated as local path."""
        storage = get_storage_from_uri("relative/path")
        assert isinstance(storage, LocalStorage)


class TestStorageIntegration:
    """Integration tests for storage backends."""

    def test_local_storage_upload_download_roundtrip(self, tmp_path: Path):
        """Test upload and download roundtrip with LocalStorage."""
        storage = LocalStorage(base_path=tmp_path / "storage")

        # Create source file
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source_file = source_dir / "test.txt"
        source_file.write_text("test data")

        # Upload
        storage.upload(source_file, "uploaded/test.txt")

        # Download to different location
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        dest_file = dest_dir / "downloaded.txt"
        storage.download("uploaded/test.txt", dest_file)

        # Verify
        assert dest_file.exists()
        assert dest_file.read_text() == "test data"

    def test_local_storage_list_and_delete(self, tmp_path: Path):
        """Test listing and deleting files."""
        storage = LocalStorage(base_path=tmp_path)

        # Create files
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.txt").write_text("b")
        (tmp_path / "dir").mkdir()
        (tmp_path / "dir" / "file3.txt").write_text("c")

        # List all files
        files = storage.list("")
        assert len(files) == 3

        # Delete one file
        storage.delete("file1.txt")

        # List again
        files = storage.list("")
        assert len(files) == 2
        assert "file1.txt" not in files

    def test_local_storage_exists_check(self, tmp_path: Path):
        """Test exists functionality."""
        storage = LocalStorage(base_path=tmp_path)

        # File doesn't exist yet
        assert not storage.exists("test.txt")

        # Create file
        (tmp_path / "test.txt").write_text("content")

        # Now it exists
        assert storage.exists("test.txt")

        # Delete it
        storage.delete("test.txt")

        # Doesn't exist again
        assert not storage.exists("test.txt")
