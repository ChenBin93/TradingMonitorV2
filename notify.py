import io
import os

import requests
from loguru import logger


class Feishu:

    def __init__(self, app_id: str, app_secret: str, chat_id: str, webhook_url: str = ""):
        self._app_id = app_id
        self._app_secret = app_secret
        self._chat_id = chat_id
        self._webhook_url = webhook_url
        self._client = None

    def _sdk_client(self):
        """懒加载 lark SDK client (避免无 SDK 环境报错)"""
        if self._client is None and self._app_id and self._app_secret:
            import lark_oapi as lark
            self._client = lark.Client.builder() \
                .app_id(self._app_id) \
                .app_secret(self._app_secret) \
                .build()
        return self._client

    def send(self, text: str):
        if not text:
            return
        lines = text.split("\n")
        title = lines[0].strip() if lines else ""
        body = lines[1:] if len(lines) > 1 else []
        post_content = [[{"tag": "text", "text": ln}] for ln in body if ln.strip()]

        if self._webhook_url:
            self._send_webhook(title, post_content)
        elif self._app_id and self._app_secret:
            self._send_sdk(title, post_content)

    def _send_webhook(self, title: str, content: list):
        try:
            resp = requests.post(self._webhook_url, json={
                "msg_type": "post",
                "content": {
                    "post": {
                        "zh_cn": {
                            "title": title,
                            "content": content,
                        }
                    }
                },
            }, timeout=10)
            if resp.status_code == 200:
                logger.info("Feishu webhook sent")
            else:
                logger.warning(f"Feishu webhook error: {resp.status_code}")
        except Exception as e:
            logger.error(f"Feishu webhook failed: {e}")

    def _send_sdk(self, title: str, content: list):
        try:
            import lark_oapi
            client = self._sdk_client()
            post = {"zh_cn": {"title": title, "content": content}}
            request = (
                lark_oapi.api.im.v1.CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    lark_oapi.api.im.v1.CreateMessageRequestBody.builder()
                    .receive_id(self._chat_id)
                    .msg_type("post")
                    .content(lark_oapi.JSON.marshal(post))
                    .build()
                )
                .build()
            )
            resp = client.im.v1.message.create(request)
            if resp.code == 0:
                logger.info("Feishu message sent")
            else:
                logger.warning(f"Feishu API error: {resp.code} {resp.msg}")
        except Exception as e:
            logger.warning(f"Feishu SDK fallback to webhook: {e}")
            self._send_webhook(title, content)

    # ── 图片推送 ─────────────────────────────────────────────
    def upload_image(self, image_path: str) -> str:
        """上传本地图片 → 返回 image_key (img_v3_xxx); 失败返回空串"""
        import lark_oapi
        client = self._sdk_client()
        if client is None:
            logger.error("Feishu image: SDK client 不可用 (缺 app_id/secret)")
            return ""
        try:
            with open(image_path, "rb") as f:
                body = lark_oapi.api.im.v1.CreateImageRequestBody.builder() \
                    .image_type("message") \
                    .image(io.BytesIO(f.read())) \
                    .build()
            request = lark_oapi.api.im.v1.CreateImageRequest.builder() \
                .request_body(body).build()
            resp = client.im.v1.image.create(request)
            if resp.code == 0 and resp.data and resp.data.image_key:
                logger.info(f"Feishu image uploaded: {resp.data.image_key} ({image_path})")
                return resp.data.image_key
            logger.warning(f"Feishu image upload failed: {resp.code} {resp.msg}")
            return ""
        except Exception as e:
            logger.error(f"Feishu image upload error: {e}")
            return ""

    def send_image(self, image_path: str, caption: str = ""):
        """上传并发送图片消息 (msg_type=image); caption 非空时先发文本再发图"""
        import lark_oapi
        if not os.path.exists(image_path):
            logger.error(f"Feishu send_image: 文件不存在 {image_path}")
            return
        if caption:
            self.send(caption)
        image_key = self.upload_image(image_path)
        if not image_key:
            return
        client = self._sdk_client()
        try:
            content = lark_oapi.JSON.marshal({"image_key": image_key})
            request = (
                lark_oapi.api.im.v1.CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    lark_oapi.api.im.v1.CreateMessageRequestBody.builder()
                    .receive_id(self._chat_id)
                    .msg_type("image")
                    .content(content)
                    .build()
                )
                .build()
            )
            resp = client.im.v1.message.create(request)
            if resp.code == 0:
                logger.info(f"Feishu image sent: {image_key}")
            else:
                logger.warning(f"Feishu image send failed: {resp.code} {resp.msg}")
        except Exception as e:
            logger.error(f"Feishu image send error: {e}")

    def send_rich(self, title: str, blocks: list[dict]):
        """富文本消息 — 一条消息内嵌文本+多图 (防刷屏)

        blocks: [{"text": str}, {"image": 本地路径}, ...] 按顺序排列
        上传图片 → 构造 post 富文本 (md 段 + img 段) → 一次发送
        """
        import lark_oapi
        client = self._sdk_client()
        if client is None:
            logger.error("Feishu send_rich: SDK client 不可用")
            return
        # 1) 上传所有图片 → image_key
        content_lines = []
        for b in blocks:
            if "image" in b and b["image"]:
                key = self.upload_image(b["image"])
                if key:
                    content_lines.append(("img", key))
                else:
                    content_lines.append(("text", f"[图上传失败: {b.get('image')}]"))
            else:
                content_lines.append(("text", b.get("text", "")))
        # 2) 构造 post 富文本 content (文本段 + 图片段交替)
        content = []
        for kind, val in content_lines:
            if kind == "text" and val.strip():
                content.append([{"tag": "text", "text": val}])
            elif kind == "img":
                content.append([{"tag": "img", "image_key": val}])
        if not content:
            return
        post = {"zh_cn": {"title": title, "content": content}}
        try:
            request = (
                lark_oapi.api.im.v1.CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    lark_oapi.api.im.v1.CreateMessageRequestBody.builder()
                    .receive_id(self._chat_id)
                    .msg_type("post")
                    .content(lark_oapi.JSON.marshal(post))
                    .build()
                )
                .build()
            )
            resp = client.im.v1.message.create(request)
            if resp.code == 0:
                logger.info(f"Feishu rich sent: {len(content_lines)} blocks")
            else:
                logger.warning(f"Feishu rich send failed: {resp.code} {resp.msg}")
        except Exception as e:
            logger.error(f"Feishu rich send error: {e}")
