from dataclasses import dataclass
import random

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
