import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import time

from common_functions import get_similarity_score, compare_items


class USE():
    def __init__(self):

        hub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        print("downloading %s" % hub_url)
        self.model = hub.load(hub_url)
        print("dowloaded 😃")
    

    def compare_texts(self, a, b):
        encodeing_a = self.encode_using_universal_sentence_encoder(a)
        encodeing_b = self.encode_using_universal_sentence_encoder(b)
        score = get_similarity_score(encodeing_a, encodeing_b)
        return score

    def encode_using_universal_sentence_encoder(self, data):
        embeddings = self.model([data])
        embeddings_normalized = tf.math.l2_normalize(embeddings)
        return embeddings_normalized
