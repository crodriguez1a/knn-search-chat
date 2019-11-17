import os
import tensorflow_hub as hub
import numpy as np
import nmslib

"""Encoder"""

USENC_4: str = os.getenv(
    'ENCODER',
    # TODO consider local download
    'https://tfhub.dev/google/universal-sentence-encoder-large/4')
DISTANCE_THRESHOLD: float = os.getenv('DISTANCE_THRESHOLD', 0.75)


class Encoder:
    __slots__ = ['module_url', '_encoder']

    def __init__(self, module_url: str):
        # initialize encoder
        self._encoder: hub.module.Module = hub.load(module_url)

    def encode(
            self,
            messages: list,) -> np.ndarray:
        # extract embeddings as numpy array
        return self._encoder(messages)["outputs"]


"""Search"""


def create_index(
        embeddings: np.ndarray,
        method: str = 'hnsw') -> nmslib.dist.FloatIndex:
    # initialize a search index
    # ref: https://github.com/nmslib/nmslib/blob/master/manual/methods.md

    # initialize a new index, using a HNSW index on Cosine Similarity
    search_index: nmslib.dist.FloatIndex = nmslib.init(
        method=method, space='cosinesimil')
    search_index.addDataPointBatch(embeddings)
    search_index.createIndex({'post': 2}, print_progress=True)

    return search_index


def search(
        query_vector: np.ndarray,
        search_index: nmslib.dist.FloatIndex,
        n_results: int = 3) -> tuple:
    # perform a knn search
    idx, dist = search_index.knnQuery(query_vector, k=n_results)
    return (idx, dist)
