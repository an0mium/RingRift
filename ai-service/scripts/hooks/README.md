# Pre-commit and CI Hooks

Python scripts for validation hooks in CI/CD pipelines and pre-commit checks.

## Scripts

| Script                         | Purpose                                     |
| ------------------------------ | ------------------------------------------- |
| `check_file_size.py`           | Validate file sizes (prevent large commits) |
| `check_resource_thresholds.py` | Check resource usage thresholds             |

## Usage

These hooks are typically run automatically by pre-commit or CI:

```bash
# Manual execution
python scripts/hooks/check_file_size.py
python scripts/hooks/check_resource_thresholds.py
```

## Configuration

See `.pre-commit-config.yaml` for hook configuration.

## See Also

- `app/coordination/safeguards.py` - Runtime resource safeguards
- `app/config/constants.py` - Threshold constants
