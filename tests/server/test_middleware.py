from server.middleware import BotMiddleWare, encode, bot, search_index

class TestMiddleware:
    def setup_class(self):
        self.bot_middleware: BotMiddleWare = BotMiddleWare(encode, bot, search_index)

    def test_post(self):
        response: str = self.bot_middleware.post("hello")
        assert response is not None
