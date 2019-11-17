import nmslib
import numpy as np

from app.app import Encoder, create_index, USENC_4
from app.chat import chat
from app.bots import QABot

Q: dict = {
    "favorite baseball team": "fav_baseball",
    "best baseball team": "fav_baseball",
    "favorite basketball team": "fav_basketball",
    "best basketball team": "fav_basketball",
    "best NBA team": "fav_basketball",
    "grew up": "hometown",
    "hometown": "hometown",
    "grow up": "hometown"
}

A: dict = {
    'fav_baseball': [
        "NY Yankees, obviously",
        "have to say...Yankees"],
    'fav_basketball': [
        "Grew up in the Jordan era...Bulls",
        "Bulls",
        "Chicago"],
    'hometown': [
        "South Norwalk, CT",
        "Connecticut",
        "Southern Connecticut right outside of NY"]}


class TestApp:
    def setup_class(self):
        encoder: Encoder = Encoder(module_url=USENC_4)
        self.encode: callable = encoder.encode
        self.qa_bot: dict = QABot(Q, A)
        keyphrase_embeddings: np.ndarray = self.encode(self.qa_bot.keyphrases)
        self.search_index: nmslib.dist.FloatIndex = create_index(
            keyphrase_embeddings)

    def test_can_known_questions(self):
        basketball: str = chat(
            "What's the best team in the NBA?",
            self.encode,
            self.qa_bot,
            self.search_index)
        hometown: str = chat(
            "Where did you grow up?",
            self.encode,
            self.qa_bot,
            self.search_index)
        baseball: str = chat(
            "What's your favorite baseball team?",
            self.encode,
            self.qa_bot,
            self.search_index)

        assert basketball in A.get('fav_basketball')
        assert hometown in A.get('hometown')
        assert baseball in A.get('fav_baseball')

    def test_unknown_answers(self):
        weather: str = chat(
            "what's the weather",
            self.encode,
            self.qa_bot,
            self.search_index)

        assert weather == "Sorry, I don't have an answer for that."
