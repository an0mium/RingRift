# Storage Module Unit Tests

This directory contains comprehensive unit tests for the `app/storage` module.

## Test Coverage

Current coverage: **61%** (38 tests, all passing)

### Covered Components

#### 1. `test_backends.py` - Storage Backend Tests (38 tests)

**TestStorageBackendInterface (2 tests)**

- Abstract class instantiation prevention
- Required method interface validation

**TestLocalStorage (21 tests)**

- Initialization and directory creation
- File upload with parent directory creation
- File download with error handling
- File listing with prefix filtering
- File existence checking
- File deletion
- Conditional download (download_if_newer)
- Path resolution and same-path handling

**TestS3Storage (1 test)**

- ImportError when boto3 not installed

**TestGCSStorage (1 test)**

- ImportError when google-cloud-storage not installed

**TestGetStorageBackend (8 tests)**

- Factory function with different backends
- Environment variable configuration
- Parameter validation
- Case-insensitive backend selection
- Explicit parameter override

**TestGetStorageFromUri (4 tests)**

- URI parsing for local files (file://)
- Plain path handling
- Unsupported scheme error handling
- Empty scheme handling

**TestStorageIntegration (3 tests)**

- End-to-end upload/download roundtrips
- List and delete operations
- Exists check workflow

## Uncovered Code

The following areas have limited coverage due to external dependencies:

- **S3Storage methods** (lines 231-296): Requires boto3
- **GCSStorage methods** (lines 321-377): Requires google-cloud-storage
- **Cloud storage factory logic** (lines 416, 421, 450-457): Requires cloud dependencies

These are tested for ImportError handling, but full integration tests would require:

- Mock AWS S3 setup (e.g., moto library)
- Mock GCS setup (e.g., gcs-emulator)

## Running Tests

```bash
# Run all storage tests
pytest tests/unit/storage/

# Run with coverage
pytest tests/unit/storage/ --cov=app/storage --cov-report=term-missing

# Run specific test class
pytest tests/unit/storage/test_backends.py::TestLocalStorage -v

# Run specific test
pytest tests/unit/storage/test_backends.py::TestLocalStorage::test_upload_copies_file -v
```

## Test Patterns

### Using tmp_path Fixture

All file system tests use pytest's `tmp_path` fixture for isolation:

```python
def test_upload_creates_parent_directories(self, tmp_path: Path):
    storage = LocalStorage(base_path=tmp_path)
    source_file = tmp_path / "source.txt"
    source_file.write_text("test content")
    storage.upload(source_file, "nested/dir/dest.txt")
    assert (tmp_path / "nested" / "dir" / "dest.txt").exists()
```

### Testing Error Conditions

```python
def test_download_raises_on_missing_file(self, tmp_path: Path):
    storage = LocalStorage(base_path=tmp_path)
    with pytest.raises(FileNotFoundError):
        storage.download("nonexistent.txt", tmp_path / "dest.txt")
```

### Testing with Environment Variables

```python
def test_uses_environment_variables(self, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_BASE_PATH", str(tmp_path))
    storage = get_storage_backend()
    assert isinstance(storage, LocalStorage)
```

## Future Enhancements

To achieve higher coverage, consider:

1. **Mock-based S3/GCS tests**: Use unittest.mock to test cloud backend logic
2. **Integration tests**: Use moto (S3) or gcs-emulator (GCS) for real integration tests
3. **Error injection tests**: Test network failures, permission errors, etc.
4. **Performance tests**: Test large file uploads, batch operations
5. **Concurrent access tests**: Test thread-safety of storage operations

## Dependencies

- `pytest`: Test framework
- `pytest-cov`: Coverage reporting
- `pathlib`: Path operations
- `unittest.mock`: Mocking for cloud backends (when needed)
