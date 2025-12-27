# Dashboard Assets

Static HTML, JavaScript, and CSS assets for the training and monitoring dashboards.

## Files

| File                      | Purpose                                |
| ------------------------- | -------------------------------------- |
| `dashboard.html`          | Main cluster monitoring dashboard      |
| `training_dashboard.html` | Training progress visualization        |
| `model_dashboard.html`    | Model comparison and metrics           |
| `model_comparison.html`   | Side-by-side model comparison          |
| `replay_viewer.html`      | Game replay visualization              |
| `replay_viewer.js`        | Replay viewer JavaScript               |
| `board_renderer.js`       | Board rendering utilities (hex/square) |

## Usage

These assets are served by the Flask dashboard server:

```bash
# Start dashboard server
python -m scripts.monitor.dashboard

# Access at http://localhost:5000
```

## See Also

- `scripts/monitor/dashboard.py` - Dashboard Flask server
- `app/routes/` - API routes for dashboard data
