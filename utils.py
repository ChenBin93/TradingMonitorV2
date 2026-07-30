# 工具：日志 + 健康检查 + 价格格式化

import math
import threading
from pathlib import Path


def fmt_price(p: float, atr: float = 1.0) -> str:
    """ATR 自适应精度：精度比 ATR 小一个数量级"""
    if p == 0:
        return "0"
    decimals = max(0, -int(math.log10(max(atr, 1e-12) / 10)))
    return f"{p:.{decimals}f}"


def round_price(p: float, atr: float = 1.0) -> float:
    """ATR 自适应精度四舍五入，返回浮点数"""
    if p == 0:
        return 0.0
    decimals = max(0, -int(math.log10(max(atr, 1e-12) / 10)))
    return round(p, decimals)


def setup_logging(level: str = "INFO", log_file: str = "logs/main.log"):
    from loguru import logger
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level, colorize=True)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level, rotation="100 MB", retention=7)


def start_health_server(port: int = 8080, checks: dict | None = None):
    """启动健康检查 HTTP 端点"""
    import http.server
    import json
    import socketserver

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                healthy = all(fn() for fn in (checks or {}).values()) if checks else True
                status = 200 if healthy else 503
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"healthy": healthy}).encode())
            else:
                self.send_response(404)
                self.end_headers()
        def log_message(self, *args): pass

    t = threading.Thread(target=lambda: socketserver.TCPServer(("", port), H).serve_forever(), daemon=True)
    t.start()
