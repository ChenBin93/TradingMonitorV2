# 飞书推送客户端

import requests
from loguru import logger


class Feishu:
    """飞书群机器人推送"""

    def __init__(self, app_id: str, app_secret: str, chat_id: str):
        self._app_id = app_id
        self._app_secret = app_secret
        self._chat_id = chat_id
        self._webhook_url = ""
        self._tenant_token = None

    def send(self, text: str):
        if not self._app_id:
            return
        if self._webhook_url:
            self._send_webhook(text)
        else:
            self._send_sdk(text)

    def _send_sdk(self, text: str):
        try:
            from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody
            from lark_oapi import Client
            client = Client.builder() \
                .app_id(self._app_id).app_secret(self._app_secret) \
                .log_level(40).build()
            req = CreateMessageRequest.builder() \
                .receive_id_type("chat_id") \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(self._chat_id)
                    .msg_type("text")
                    .content(f'{{"text":"{text}"}}')
                    .build()
                ).build()
            client.im.v1.message.create(req)
            logger.debug("Feishu message sent")
        except Exception as e:
            logger.warning(f"Feishu SDK failed: {e}, trying webhook fallback")
            self._send_webhook(text)

    def _send_webhook(self, text: str):
        try:
            requests.post(self._webhook_url, json={
                "msg_type": "text",
                "content": {"text": text},
            }, timeout=10)
        except Exception as e:
            logger.error(f"Feishu webhook failed: {e}")
