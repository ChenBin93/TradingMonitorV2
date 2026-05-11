#!/bin/bash
# TradingMonitor V2 启动脚本

cd /root/workspace/project/TradingMonitor/v2

# 先杀掉旧进程
pkill -f "python3 main.py" 2>/dev/null
sleep 2

# 用 nohup 脱离终端启动，不受 shell 退出影响
nohup python3 main.py > /tmp/v2.log 2>&1 &
PID=$!

echo "V2 started (PID: $PID)"
echo "Log: logs/main.log"
echo "External log: /tmp/v2.log"
echo "Check: tail -f logs/main.log"
echo "Kill: kill $PID"
