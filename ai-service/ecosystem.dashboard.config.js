/**
 * PM2 Ecosystem Configuration for RingRift AI Dashboard
 *
 * Manages the Flask-based AI training dashboard server that provides:
 * - Elo progress visualization
 * - Training metrics
 * - Cluster status
 *
 * Usage:
 *   pm2 start ecosystem.dashboard.config.js
 *   pm2 restart ringrift-ai-dashboard
 *   pm2 logs ringrift-ai-dashboard
 *
 * Deployment:
 *   1. Copy ai-service/ to production server
 *   2. Install dependencies: pip install flask flask-cors
 *   3. Start with PM2: pm2 start ecosystem.dashboard.config.js
 *   4. Reload nginx: sudo nginx -t && sudo nginx -s reload
 *   5. Access at: https://ringrift.ai/ai-dashboard/progress_dashboard.html
 */
module.exports = {
  apps: [
    {
      name: 'ringrift-ai-dashboard',
      script: 'scripts/dashboard_server.py',
      interpreter: 'python3',
      cwd: '/home/ubuntu/ringrift/ai-service',
      args: '--port 8080 --host 127.0.0.1',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        PYTHONPATH: '.',
        FLASK_ENV: 'production',
      },
      error_file: '/home/ubuntu/ringrift/ai-service/logs/dashboard-error.log',
      out_file: '/home/ubuntu/ringrift/ai-service/logs/dashboard-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
    },
  ],
};
