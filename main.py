import tensorflow as tf
import tensorflow_hub as hub
import sentencepiece as spm
import bert
import os
import csv
import numpy as np


def main():
    header, argos_items_data = readCsv("./MOCK_DATA_small.csv")
    test_sainsbury_item = ["sainsburys sku", "title", "description"]

    a =  USE(test_sainsbury_item[1:], argos_items_data )

    print(np.array(a.scores))
    

    # ALBERT(argos, sainsbury)

def readCsv(file_path):
    data = []
    with open(file_path,) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        header = None
        for row in csvReader:
            data.append(row)
    header = data.pop(0)
    return header, data

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
        datapoint_score  = []
        for i ,column in enumerate(datapoint):
            datapoint_score.append(1.0 - cosine_similarity(a[i], column))
        scores.append(datapoint_score)
    return scores

class USE():
    def __init__(self, original_text, comparison_items, batch_size=10000):
       
        self.model = tf.keras.models.load_model('models/use/')

        original_text_encoding = self.encode_all_columns(original_text)
        
        self.results = []
        product_matrix = np.array(comparison_items)

        datapoints = product_matrix[:,1:len(product_matrix[0])]

       
        output =self.encode_all_rows(datapoints)
        
        self.scores = get_similarity_scores(original_text_encoding, output)
        ids = np.array(comparison_items)[:,0]
        self.results = np.column_stack([ids, self.scores])
        print(self.results)

    
    def encode_all_columns(self,datapoints):
        output = []
        for column in datapoints:
            output.append(
                self.encode_using_universal_sentence_encoder(
                    column,self.model))
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
        model_dir = bert.fetch_google_albert_model(model_name, "models")
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
        token_ids_b= self.preprocess_input(
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
        processed_text =bert.albert_tokenization.preprocess_text(text, lower=lower)
        return bert.albert_tokenization.encode_ids(smp_model, processed_text)
# https://github.com/kpe#/bert-for-tf2

    def predict_result(self, token_ids):
        return self.model.predict(
            token_ids, batch_size=1, use_multiprocessing=True)
        



if __name__ == "__main__":
    main()
