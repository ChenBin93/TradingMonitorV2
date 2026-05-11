# 飞书推送客户端

import requests
from loguru import logger


class Feishu:

    def __init__(self, app_id: str, app_secret: str, chat_id: str, webhook_url: str = ""):
        self._app_id = app_id
        self._app_secret = app_secret
        self._chat_id = chat_id
        self._webhook_url = webhook_url

    def send(self, text: str):
        if not text:
            return
        if self._webhook_url:
            self._send_webhook(text)
        elif self._app_id and self._app_secret:
            self._send_sdk(text)

    def _send_webhook(self, text: str):
        try:
            resp = requests.post(self._webhook_url, json={
                "msg_type": "text",
                "content": {"text": text},
            }, timeout=10)
            if resp.status_code == 200:
                logger.info("Feishu webhook sent")
            else:
                logger.warning(f"Feishu webhook error: {resp.status_code}")
        except Exception as e:
            logger.error(f"Feishu webhook failed: {e}")

    def _send_sdk(self, text: str):
        try:
            import lark_oapi as lark
            client = lark.Client.builder().app_id(self._app_id).app_secret(self._app_secret).build()
            request = (
                lark.api.im.v1.CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    lark.api.im.v1.CreateMessageRequestBody.builder()
                    .receive_id(self._chat_id)
                    .msg_type("text")
                    .content(lark.JSON.marshal({"text": text}))
                    .build()
                )
                .build()
            )
            resp = client.im.v1.message.create(request)
            if resp.code == 0:
                logger.info("Feishu message sent")
            else:
                logger.warning(f"Feishu API error: {resp.msg}")
        except Exception as e:
            logger.error(f"Feishu SDK failed: {e}")
