#!/bin/bash
# 自动更新监控币种列表并重启 V2
# 建议每周执行一次: crontab -e → 0 8 * * 1 /root/workspace/project/TradingMonitor/v2/update_watchlist.sh

cd /root/workspace/project/TradingMonitor/v2

echo "[$(date)] 开始筛选币种..."
python3 select_symbols.py --pool 100 --corr 0.8 --write-config

echo "[$(date)] 重启 V2..."
kill $(ps aux | grep "python3 main" | grep -v grep | awk '{print $2}') 2>/dev/null
sleep 3
bash start.sh

echo "[$(date)] 完成"
