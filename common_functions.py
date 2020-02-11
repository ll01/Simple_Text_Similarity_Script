
import tensorflow as tf
import numpy as np

import os
import time




def cosine_similarity(a, b):

    # https://en.wikipedia.org/wiki/Cosine_similarity
    # reduce sum multiply calculates the dot product
    # we normalised the vectors so its just a divide
    # by one (which simplifies) and we are serching
    # for theta so we need to use inverse cos
    cosine_similarity = tf.tensordot(a, b, 2)
    # normalizing between 1 and -1
    clip_cosine_similarities = tf.clip_by_value(cosine_similarity, -1.0, 1.0)
    return tf.acos(clip_cosine_similarities)


def get_similarity_score(a, b):
    return 1.0 - cosine_similarity(a, b)


def get_similarity_scores(a, datapoints):
    scores = []
    for datapoint in datapoints:
        datapoint_score = []
        for i, column in enumerate(datapoint):
            if (tf.is_tensor(column)):
                score = 1.0 - cosine_similarity(a[i], column)
            else:
                score = tf.constant("n/a")
            datapoint_score.append(score.numpy())
        scores.append(datapoint_score)
    return scores

def compare_items(original_item, comparison_items, encoder):
    start_time = time.time()
    original_text_encoding = encode_row(
        original_item[1:],encoder)

    comparison_text = [row[1:] for row in comparison_items]
    comparison_ids = np.array(comparison_items)[:, 0]
    output = encode_dataset(
        comparison_text, encoder)

    scores = get_similarity_scores(original_text_encoding, output)

    end_time = time.time()
    print("comparison of {} items ran for {:0.2f}s".format(
        len(comparison_items),  end_time - start_time))

    return np.column_stack([comparison_ids, scores])


def encode_dataset(datapoints, encoder):
        output = []
        for row in datapoints:
            output.append(encode_row(row, encoder))
        return output

def encode_row(row, encoder):
    output = []
    for column in row:
        if (column != ""):
            output.append(
                encoder(column))
        else:
            output.append(None)
    return output
