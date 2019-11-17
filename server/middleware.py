import os
import yaml
import nmslib
import numpy as np

from app.app import Encoder, create_index, USENC_4
from app.chat import chat
from app.bots import QABot


def load_bot_meta(path) -> tuple:
    with open(path, 'r') as stream:
        try:
            data: dict = yaml.safe_load(stream)
            return (data.get('questions'), data.get('answers'))
        except yaml.YAMLError as e:
            raise e


class BotMiddleWare:
    __slots__ = ['encode', 'bot', 'search_index']

    def __init__(self, encode: callable, bot: QABot,
                 search_index: nmslib.dist.FloatIndex):
        self.encode = encode
        self.bot = bot
        self.search_index = search_index

    def post(self, message: str):
        return chat(
            message,
            self.encode,
            self.bot,
            self.search_index)


# TODO REMOVE
tmp_path: str = os.getcwd() + '/data/qa_bot.yml'

# initialize bot
path: str = os.getenv('BOT_DATA', tmp_path)

encoder: Encoder = Encoder(module_url=USENC_4)
encode: callable = encoder.encode

Q, A = load_bot_meta(path)
bot: QABot = QABot(Q, A)

# initialize search index
keyphrase_embeddings: np.ndarray = encode(bot.keyphrases)
search_index: nmslib.dist.FloatIndex = create_index(keyphrase_embeddings)
