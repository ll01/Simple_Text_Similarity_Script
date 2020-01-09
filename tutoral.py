import tensorflow as tf
import tensorflow_hub as hub
import sentencepiece as spm
import bert
import os


def main():
    sainsbury = """With its modern ridged texture design and high gloss finish, this Breville® Impressions Vanilla Cream 4 Slice Toaster will make a style statement in any kitchen. Packed with handy features to create your perfect slice, variable width slots accommodate different bread sizes, crumpets, bagels etc, while the high-lift facility means no more burnt fingers when removing smaller toast slices. Illuminated control buttons highlight the defrost, reheat and cancel functions for quick and easy use and variable browning controls to ensure every slice is toasted to your liking."""
    argos = """With its modern ridged texture design and high gloss finish, this Breville Impressions Vanilla Cream 4 Slice Toaster will make a style statement in any kitchen. There to help at breakfast, lunch or dinner, this Impressions Toaster is packed with handy features to create your perfect slice.

        Variable width slots take different bread sizes, crumpets & bagels; high-lift makes removing smaller slices easy; and variable browning means each slice is just how you like it. Defrost, reheat & cancel controls are illuminated for quick & easy use. """
    USE(argos, sainsbury)
    # ALBERT(argos, sainsbury)

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


class USE():
    def __init__(self, first_text, second_text):
        self.hub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        model = hub.load(self.hub_url)
        a = self.encode_using_universal_sentence_encoder(first_text, model)
        b = self.encode_using_universal_sentence_encoder(second_text, model)
        print(get_similarity_score(a, b))

    def encode_using_universal_sentence_encoder(self, text, model_from_hub):
        embeddings = model_from_hub([text])
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

        spm_model = os.path.join(model_dir, "assets", "30k-clean.model")
        sp = spm.SentencePieceProcessor()
        sp.load(spm_model)

        do_lower_case = True

        token_ids_a = self.preprocess_input(
            spm_model, first_text, do_lower_case)
        token_ids_b= self.preprocess_input(
            spm_model, second_text, do_lower_case)

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
