import tensorflow as tf
import tensorflow_hub as hub


def main():
    texts = [
        "my name is David",
        "David is the name I was given"
    ]
    print("similar texts")
    how_similar_USE(texts[0], texts[1])

    texts_not_similar = [
        "Boats are really cool",
        "Dude where are we ?"
    ]
    print("not similar texts")
    how_similar_USE(texts_not_similar[0], texts_not_similar[1])


def encode_using_universal_sentence_encoder(text, model_from_hub):
    embeddings = model_from_hub([text])
    embeddings_normalized = tf.math.l2_normalize(embeddings)
    return embeddings_normalized


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


def how_similar_USE(first_text, second_text):
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    a = encode_using_universal_sentence_encoder(first_text, model)
    b = encode_using_universal_sentence_encoder(second_text, model)
    print(get_similarity_score(a, b))


if __name__ == "__main__":
    main()
