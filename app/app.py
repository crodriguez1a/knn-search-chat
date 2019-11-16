import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import nmslib

"""Encoder"""

# initialize encoder
USENC_4: str = "https://tfhub.dev/google/universal-sentence-encoder-large/4"
encoder: hub.module.Module = hub.load(USENC_4)

# extract embeddings as numpy array
def encode(messages: list, encoder: hub.module.Module=encoder) -> np.ndarray:
    return encoder(messages)["outputs"]

"""Search"""

# initialize a search index
def create_index(embeddings: np.ndarray, method: str='hnsw') -> nmslib.dist.FloatIndex:
    # ref: https://github.com/nmslib/nmslib/blob/master/manual/methods.md

    # initialize a new index, using a HNSW index on Cosine Similarity
    search_index: nmslib.dist.FloatIndex = nmslib.init(method=method, space='cosinesimil')
    search_index.addDataPointBatch(embeddings)
    search_index.createIndex({'post': 2}, print_progress=True)

    return search_index

# perform a knn search
def search(query_vector: np.ndarray, search_index: nmslib.dist.FloatIndex, n_results:int = 3) -> tuple:
    idx, dist = search_index.knnQuery(query_vector, k=n_results)
    return (idx, dist)
