#!/bin/bash
# TradingMonitor V2 启动脚本
cd /root/workspace/project/TradingMonitor/v2

# 杀掉所有旧进程
for pid in $(ps aux | grep "python3 .*main.py" | grep -v grep | awk '{print $2}'); do
    kill -9 $pid 2>/dev/null
done
sleep 2

# 用 setsid 脱离进程组启动 (避免外层 shell 超时连带杀进程)
setsid nohup python3 main.py > /tmp/v2.log 2>&1 < /dev/null &
PID=$!

echo "V2 started (PID: $PID)"
echo "Log: logs/main.log"
echo "Kill: kill $PID"
