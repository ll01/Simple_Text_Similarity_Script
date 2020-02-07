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
import transformers


def main():

    args = set_up_args()

    header, argos_items_data = readCsv(args.argos)
    print(header)
    _, sainsbury_items_data = readCsv(args.sainsbury)
    comparison_engine = USE()

    item_start = 0

    batch_start = 0

    batch_count = math.ceil(len(argos_items_data) / args.batch_size)

    sainsbury_items_data = list(enumerate(sainsbury_items_data))[item_start:]

    for sainsbury_item_index, item in sainsbury_items_data:
        for batch in range(batch_start, batch_count):
            print("sainsbury\'s id {} item {} batch {} out of {}".format(
                item[0], sainsbury_item_index, batch + 1, batch_count))
            start_index = (batch * args.batch_size)
            end_index = min((start_index+args.batch_size),
                            len(argos_items_data))
            print("start_index: {}, end_index: {}".format(start_index, end_index))
            comparison_engine.compare_items(
                item, argos_items_data[start_index:end_index])

            file_path = os.path.join(args.output, "{}.csv".format(item[0]))
            save_results(comparison_engine.scores, file_path)
            write_progress(sainsbury_item_index, batch)
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


class USE():
    def __init__(self):

        hub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.model = hub.load(hub_url)
        self.scores = None

    def compare_items(self, original_item, comparison_items):
        start_time = time.time()
        original_text_encoding = self.encode_all_columns(original_item[1:])

        comparison_text = [row[1:] for row in comparison_items]
        comparison_ids = np.array(comparison_items)[:, 0]
        output = self.encode_all_rows(comparison_text)

        self.scores = get_similarity_scores(original_text_encoding, output)
        self.scores = np.column_stack([comparison_ids, self.scores])

        end_time = time.time()
        print("comparison of {} items ran for {:0.2f}s".format(
            len(comparison_items),  end_time - start_time))

    def compare_texts(self, a, b):
        encodeing_a = self.encode_using_universal_sentence_encoder(
            a, self.model)
        encodeing_b = self.encode_using_universal_sentence_encoder(
            b, self.model)
        score = get_similarity_score(encodeing_a, encodeing_b)
        return score

    def encode_all_columns(self, datapoints):
        output = []
        for column in datapoints:
            if (column != ""):
                output.append(
                    self.encode_using_universal_sentence_encoder(
                        column, self.model))
            else:
                output.append(None)
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
        model_name = 'albert-base-v2'

        self.tokenizer = transformers.AlbertTokenizer.from_pretrained(
            model_name)

        hub_url = "https://tfhub.dev/google/albert_base/3"
        print("downloading %s" % hub_url)
        self.model = transformers.TFAlbertModel.from_pretrained(model_name)
        print("dowloaded 😃")

        truth = "I like my self."
        max_text_size = 128
        embeddings_a = self.get_embeddings([first_text], max_text_size)
        embeddings_b = self.get_embeddings([second_text], max_text_size)
        print(get_similarity_score(embeddings_a, embeddings_b))

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

    def get_embeddings(self, input_list, max_text_size=512):
        token_ids, input_mask, segment_ids = self.preprocess_inputs(
            input_list,  max_text_size)
        pooled_output = self.model([token_ids, input_mask, segment_ids])[0]
        output = tf.keras.layers.GlobalAveragePooling1D()(pooled_output)
        return tf.math.l2_normalize(output)

    def build_network(self, albert_layer, max_text_size):
        input_ids = tf.keras.layers.Input(
            shape=(max_text_size,), dtype='int32')
        # also known as mask
        token_type_ids = tf.keras.layers.Input(
            shape=(max_text_size,), dtype='int32')

        output = albert_layer([input_ids, token_type_ids])
        output = tf.keras.layers.GlobalAveragePooling1D()(output)
        model = tf.keras.Model(
            inputs=[input_ids, token_type_ids], outputs=output)
        model.build(input_shape=[(None, max_text_size), (None, max_text_size)])
        return model

    def pad_tokens_and_generate_mask(self, tokens, max_text_size):
        mask = []
        for _ in range(len(tokens)):
            mask.append(1)
        for _ in range(max_text_size - len(tokens)):
            tokens.append(0)
            mask.append(0)
        return tokens, mask

    def predict_result(self, token_ids, mask, max_text_size):
        ids = np.array([token_ids])
        mask = np.array([mask])
        # x = tf.reshape(x, shape=(-1, 128))
        return tf.math.l2_normalize(self.model.predict(
            [ids, mask], use_multiprocessing=True))


if __name__ == "__main__":
    os.environ["TFHUB_CACHE_DIR"] = './models'
    main()
