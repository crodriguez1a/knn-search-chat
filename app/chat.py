import time
import sys

import numpy as np
import nmslib

from app.app import search, DISTANCE_THRESHOLD
from app.bots import QABot  # TODO abstract base class for type hinting

"""Chat Interface"""


def bubbles(pause: int):
    # credit https://gist.github.com/Y4suyuki/6805818
    animation = "|/-\\"

    for i in range(pause):
        time.sleep(0.1)
        sys.stdout.write("\r" + animation[i % len(animation)])
        sys.stdout.flush()


def chat(message: str, encode: callable, bot: QABot,
         search_index: nmslib.dist.FloatIndex):

    # encode query
    vectory_query: np.ndarray = encode([message])

    # get search results
    idx, dist = search(vectory_query, search_index)

    # traverse to the first answer
    if idx.any():
        if dist[0] < DISTANCE_THRESHOLD:
            return bot.answers_index(idx[0])
        else:
            return "Sorry, I don't have an answer for that."  # TODO no english
