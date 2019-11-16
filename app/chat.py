import time
import sys

import numpy as np
from dataclasses import dataclass
import random
import nmslib

from app.app import search, encode

"""Bot"""


@dataclass
class QABot:
    queries: dict
    answers: dict

    @property
    def keyphrases(self) -> list:
        return list(self.queries.keys())

    def answers_index(self, idx: int) -> str:
        # match an index to a corresponding key-phrase
        keyphrase: str = self.keyphrases[idx]
        # use the plain text key-phrase to map to an answer
        answer_key: str = self.queries[keyphrase]
        # mapped answer
        answer: str = self.answers[answer_key]
        # randomize the answer for variety
        return random.choice(answer)


"""Chat Interface"""


def _bubbles(pause: int):
    # credit https://gist.github.com/Y4suyuki/6805818
    animation = "|/-\\"

    for i in range(pause):
        time.sleep(0.1)
        sys.stdout.write("\r" + animation[i % len(animation)])
        sys.stdout.flush()


def chat(message: str, bot: QABot, search_index: nmslib.dist.FloatIndex):
    # delay animation
    _bubbles(5)

    # encode query
    vectory_query: np.ndarray = encode([message])

    # get search results
    idx, dist = search(vectory_query, search_index)

    # traverse to the first answer
    if idx.any():
        if dist[0] < 0.75:
            return bot.answers_index(idx[0])
        else:
            return "Sorry, I don't have an answer for that."  # TODO no english
