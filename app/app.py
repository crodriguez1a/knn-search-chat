import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import random
import nmslib
import sentencepiece as spm

# tensorflow hub does not support eager execution # https://github.com/tensorflow/hub/issues/124
# tf.compat.v1.disable_eager_execution()
# tf = tf.compat.v1

USENC_4 = "https://tfhub.dev/google/universal-sentence-encoder-large/4"
# USENC_2 = "https://tfhub.dev/google/universal-sentence-encoder/2"
# USENC_LITE2 = "https://tfhub.dev/google/universal-sentence-encoder-lite/2"

def load_encoder(module_url:str) -> hub.module.Module:
    # ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
    # import as tf module
    return hub.load(module_url)

def encode(embed: hub.module.Module, messages: list) -> np.ndarray:
    # TODO cache anything previously encoded for this turn
    return embed(messages)['outputs']

def create_index(embeddings: np.ndarray, method: str='hnsw') -> nmslib.dist.FloatIndex:
    """
    Ref: https://github.com/nmslib/nmslib/blob/master/manual/methods.md
    """
    # initialize a new index, using a HNSW index on Cosine Similarity
    search_index: nmslib.dist.FloatIndex = nmslib.init(method=method, space='cosinesimil')
    search_index.addDataPointBatch(embeddings)
    search_index.createIndex({'post': 2}, print_progress=True)

    return search_index

def search(query_vector: np.ndarray, n_results:int = 3) -> tuple:
    idx, dist = search_index.knnQuery(query_vector, k=n_results)
    return (idx, dist)

def update_search_index(embed, queries: list) -> nmslib.dist.FloatIndex:
    # encode queries
    print('Encoding bot data...')
    query_embeddings: np.ndarray = encode(embed, queries)

    return create_index(query_embeddings)

if __name__ == "__main__":
    # sample bot
    qna: dict = {
        'queries' : {
            "favorite baseball team" : "fav_baseball",
            "best baseball team" : "fav_baseball"
          },

          'answers' : {
              'fav_baseball': ["NY Yankees, obviously", "have to say...Yankees"]
          }
    }

    # TODO move

    # load encoder module
    print("Loading encoder...")
    embed: hub.module.Module = load_encoder(USENC_4)

    # assemble possible queries
    queries: list = list(qna['queries'].keys())

    # encode queries
    print('Encoding bot data...')
    query_embeddings: np.ndarray = encode(embed, queries)

    # create search index
    search_index: nmslib.dist.FloatIndex = create_index(query_embeddings)

    # TEMP
    idx, dist = search(encode(embed, ["What's your favorite baseball team"]))

    if idx.any():
        search_result: str = queries[idx[0]]
        answer_key: str = qna['queries'][search_result]
        answer: str = qna['answers'][answer_key]
        print(random.choice(answer))
