#!/usr/bin/env bash
#
# Whale API Server — PM2 管理脚本
# 使用方式:
#   bash scripts/manage_server.sh start   启动服务
#   bash scripts/manage_server.sh stop    停止服务
#   bash scripts/manage_server.sh restart 重启服务
#   bash scripts/manage_server.sh status  查看状态
#   bash scripts/manage_server.sh logs    查看日志
#

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="whale-api"
ECOSYSTEM="$PROJECT_DIR/ecosystem.config.js"

ensure_pm2() {
  if ! command -v pm2 &>/dev/null; then
    echo "📦 Installing pm2..."
    npm install -g pm2
  fi
}

case "${1:-help}" in
  start)
    ensure_pm2
    echo "🚀 Starting Whale API Server on port 7999..."
    cd "$PROJECT_DIR"
    pm2 start "$ECOSYSTEM" --only "$APP_NAME"
    pm2 save
    echo "✅ Whale API Server started (PID: $(pm2 pid $APP_NAME))"
    echo "   API:    http://localhost:7999/api/health"
    echo "   Docs:   http://localhost:7999/docs"
    ;;
  stop)
    ensure_pm2
    echo "🛑 Stopping Whale API Server..."
    pm2 stop "$APP_NAME" 2>/dev/null || true
    echo "✅ Stopped"
    ;;
  restart)
    ensure_pm2
    echo "🔄 Restarting Whale API Server..."
    cd "$PROJECT_DIR"
    pm2 restart "$ECOSYSTEM" --only "$APP_NAME"
    echo "✅ Restarted"
    ;;
  status)
    ensure_pm2
    pm2 show "$APP_NAME" 2>/dev/null || echo "❌ Whale API Server is not running"
    ;;
  logs)
    ensure_pm2
    pm2 logs "$APP_NAME" --lines 50
    ;;
  help|*)
    echo "Usage: $0 {start|stop|restart|status|logs}"
    exit 1
    ;;
esac
