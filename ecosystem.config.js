module.exports = {
  apps: [{
    name: "whale-api",
    script: "whale_video/server.py",
    interpreter: "python3",
    cwd: __dirname,
    args: ["--port", "7999", "--host", "0.0.0.0"],
    env: {
      PYTHONUNBUFFERED: "1",
    },
    // Auto-restart on crash
    max_restarts: 10,
    min_uptime: "10s",
    restart_delay: 5000,
    // Logging
    log_date_format: "YYYY-MM-DD HH:mm:ss Z",
    error_file: "logs/whale-api-error.log",
    out_file: "logs/whale-api-out.log",
    merge_logs: true,
    // Resource limits
    max_memory_restart: "32G",
  }],
};
