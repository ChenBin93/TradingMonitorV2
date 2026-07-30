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
            import lark_oapi as lark
            client = lark.Client.builder().app_id(self._app_id).app_secret(self._app_secret).build()
            post = {"zh_cn": {"title": title, "content": content}}
            request = (
                lark.api.im.v1.CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    lark.api.im.v1.CreateMessageRequestBody.builder()
                    .receive_id(self._chat_id)
                    .msg_type("post")
                    .content(lark.JSON.marshal(post))
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
