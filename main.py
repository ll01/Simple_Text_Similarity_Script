import tensorflow as tf
import tensorflow_hub as hub
import sentencepiece as spm
import bert
import os
import csv
import numpy as np
import argparse
import math
import time


def main():

    args = set_up_args()

    header, argos_items_data = readCsv(args.argos)
    print(header)
    _, sainsbury_items_data = readCsv(args.sainsbury)
    comparison_engine = USE()

    item_start = 0
    batch_start = 0

    batch_count = math.ceil(len(argos_items_data) / args.batch_size)

    for item_index, item in enumerate(sainsbury_items_data, item_start):
        for batch in range(batch_start, batch_count):
            print("item {} batch {} out of {}".format(
                item_index + 1, batch + 1, batch_count))
            start_index = (batch * args.batch_size)
            end_index = min((start_index+args.batch_size),
                            len(argos_items_data))
            print("start_index: {}, end_index: {}".format(start_index, end_index))
            comparison_engine.compare_items(
                item, argos_items_data[start_index:end_index])
            file_path = os.path.join(args.output, "{}.csv".format(item[0]))
            save_results(comparison_engine.scores, file_path)
            write_progress(item_index, batch)
        batch_start = 0


def set_up_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--argos', help='argos items csv', required=False,
        default="./argos.csv")
    parser.add_argument(
        '-s', '--sainsbury',
        help='sainsbury\'s items csv this is checked against the argos items'
        'to find maches', required=False, default="./sainsburys.csv")
    parser.add_argument('-o', '--output',
                        help='where the output directory', required=False,
                        default="./results/")
    parser.add_argument('-b', '--batch_size',
                        help='amount of argos items to compare in each batch',
                        required=False, type=int, default=10000)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    return args


def readCsv(file_path, delim=","):
    data = []
    with open(file_path,) as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter=delim)
        header = None
        for row in csvReader:
            if len(row) == 3:
                data.append(row)
    header = data.pop(0)
    return header, data


def save_results(data, file_path):
    with open(file_path, "a+", newline='') as csvDataFile:
        for row in data:
            csvWriter = csv.writer(csvDataFile)
            csvWriter.writerow(row)


def write_progress(item, batch):
    with open("progress.txt", "w") as progress_file:
        progress_file.write("{}\n{}\n".format(item, batch))


def cosine_similarity(a, b):

    # https://en.wikipedia.org/wiki/Cosine_similarity
    # reduce sum multiply calculates the dot product
    # we normalised the vectors so its just a divide
    # by one (which simplifies) and we are serching
    # for theta so we need to use inverse cos
    cosine_similarity = tf.tensordot(a[0], b[0], 1)
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
            datapoint_score.append(1.0 - cosine_similarity(a[i], column))
        scores.append(datapoint_score)
    return scores


class USE():
    def __init__(self):

        hub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.model = hub.load(hub_url)
        self.scores = None

    def compare_items(self, original_item, comparison_items):
        start_time = time.time()
        original_text_encoding = self.encode_all_columns(original_item[1:])

        comparison_text = [row[1:3] for row in comparison_items]
        comparison_ids = np.array(comparison_items)[:, 0]
        output = self.encode_all_rows(comparison_text)

        self.scores = get_similarity_scores(original_text_encoding, output)
        self.scores = np.column_stack([comparison_ids, self.scores])

        end_time = time.time()
        print("comparison of {} items ran for {:0.2f}s".format(
            len(comparison_items),  end_time - start_time))

    def encode_all_columns(self, datapoints):
        output = []
        for column in datapoints:
            output.append(
                self.encode_using_universal_sentence_encoder(
                    column, self.model))
        return output

    def encode_all_rows(self, datapoints):
        output = []
        for row in datapoints:
            output.append(self.encode_all_columns(row))
        return output

    def encode_using_universal_sentence_encoder(self, data, model_from_hub):
        embeddings = model_from_hub([data])
        embeddings_normalized = tf.math.l2_normalize(embeddings)
        return embeddings_normalized


class ALBERT():
    def __init__(self, first_text, second_text):
        model_name = "albert_base_v2"
        model_dir = bert.fetch_google_albert_model(model_name, ".models")
        model_ckpt = os.path.join(model_dir, "model.ckpt-best")
        model_params = bert.albert_params(model_dir)

        albert_layer = bert.BertModelLayer.from_params(
            model_params, name="albert")
        max_text_size = 128
        self.model = self.build_network(albert_layer, max_text_size)
        bert.load_albert_weights(albert_layer, model_ckpt)

        spm_model = os.path.join(model_dir, "30k-clean.model")
        sp = spm.SentencePieceProcessor()
        sp.load(spm_model)

        do_lower_case = True

        token_ids_a = self.preprocess_input(
            sp, first_text, do_lower_case)
        token_ids_b = self.preprocess_input(
            sp, second_text, do_lower_case)

        text_prediction_a = self.predict_result(token_ids_a)
        text_prediction_b = self.predict_result(token_ids_b)
        print(get_similarity_score(text_prediction_a, text_prediction_b))

    def build_network(self, albert_layer, max_text_size):
        input_ids = tf.keras.layers.Input(
            shape=(max_text_size,), dtype='int32')
        output = albert_layer(input_ids)
        model = tf.keras.Model(inputs=input_ids, outputs=output)
        model.build(input_shape=(None, max_text_size))
        return model

    def preprocess_input(self, smp_model, text, lower=True):
        processed_text = bert.albert_tokenization.preprocess_text(
            text, lower=lower)
        return bert.albert_tokenization.encode_ids(smp_model, processed_text)
# https://github.com/kpe#/bert-for-tf2

    def predict_result(self, token_ids):
        return self.model.predict(
            token_ids, batch_size=1, use_multiprocessing=True)


if __name__ == "__main__":
    os.environ["TFHUB_CACHE_DIR"] = './models'
    main()
