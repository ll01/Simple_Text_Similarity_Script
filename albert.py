import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import transformers

import time

from common_functions import encode_row, encode_dataset, get_similarity_score,\
    get_similarity_scores


class ALBERT():
    def __init__(self, max_text_size=128):
        model_name = 'albert-base-v2'

        self.tokenizer = transformers.AlbertTokenizer.from_pretrained(
            model_name)

        hub_url = "https://tfhub.dev/google/albert_base/3"
        print("downloading %s" % hub_url)
        self.model = transformers.TFAlbertModel.from_pretrained(model_name)
        print("dowloaded 😃")
        self.max_text_size = max_text_size


    def compare_texts(self):
         pass 

    def preprocess_inputs(self, input_list, max_text_size):
        input_ids = np.empty((len(input_list), max_text_size), dtype=int)
        attention_mask = np.empty((len(input_list), max_text_size), dtype=int)
        # also known as segment ids
        token_type_ids = np.empty((len(input_list), max_text_size), dtype=int)

        for index, input in enumerate(input_list):
            tokenized_input = self.tokenizer.encode_plus(
                input, max_length=max_text_size,
                pad_to_max_length=True)
            input_ids[index] = tokenized_input["input_ids"]
            attention_mask[index] = tokenized_input["attention_mask"]
            token_type_ids[index] = tokenized_input["token_type_ids"]
        return input_ids, attention_mask, token_type_ids

    def get_embeddings(self, input_list):
        token_ids, input_mask, segment_ids = self.preprocess_inputs(
            input_list,  self.max_text_size)
        pooled_output = self.model([token_ids, input_mask, segment_ids])[0]
        output = tf.keras.layers.GlobalAveragePooling1D()(pooled_output)
        return tf.math.l2_normalize(output)