import random
import collections

import nltk
import numpy as np
import regex as re
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from matplotlib import pyplot as plt


nltk.download('punkt')
nltk.download('stopwords')


def prepare_text(s):
    text = s.replace("\n", " ").replace("-", " ").lower()
    text = re.sub('[–?!@#$;—:,*~()‘”“]', '', text)
    word_array = [word for word in text.split() if word not in stopwords.words('portuguese')]
    return " ".join(word_array)


def text_to_tokens(text):
    sentence = text.split('.')
    sentence = list(filter(None, sentence))

    pt_stp_words = stopwords.words('portuguese')
    tokens = [nltk.word_tokenize(words) for words in sentence]
    for i, token in enumerate(tokens):
        tokens[i] = [word for word in token if word not in pt_stp_words]
    return tokens


def get_most_frequent_words(text, quantity):
    split_it = text.replace(".", "").split()
    counter = collections.Counter(split_it)
    return counter.most_common(quantity)


def show_similarities(model, top_words):
    count = len(top_words)
    word_array = []
    print("\n")
    while count > 0:
        word_1 = word_2 = ' '
        while len([word for word in word_array if word_1 in word and word_2 in word]) > 0 or word_1 == word_2:
            word_1 = top_words[random.randint(0, len(top_words) - 1)][0]
            word_2 = top_words[random.randint(0, len(top_words) - 1)][0]

        print("The similarity between '" + word_1 + "' and '" + word_2 + "' is: ", end="")
        print(model.wv.similarity(word_1, word_2) * 100, end="%\n")
        word_array.append((word_1, word_2))
        count -= 1
    print("\n")


def plot_pca(model, annotate=True):
    x = model.wv[model.wv.key_to_index]
    all_words = list(model.wv.key_to_index)

    if annotate:
        print("\n")
        print(x, end="\n")

    x_corr = pd.DataFrame(x).corr()
    values, vectors = np.linalg.eig(x_corr)

    args = (-values).argsort()
    vectors = vectors[:, args]
    new_vectors = vectors[:, :2]
    new_x = np.dot(x, new_vectors)

    plt.figure(figsize=(13, 7))
    plt.scatter(np.real(new_x[:, 0]), np.real(new_x[:, 1]), linewidths=10, color='green')
    plt.xlabel("PC1", size=15)
    plt.ylabel("PC2", size=15)
    plt.title("PCA", size=20)
    if annotate:
        for i, word in enumerate(all_words):
            plt.annotate(word, xy=(np.real(new_x[i, 0]), np.real(new_x[i, 1])))

    plt.show()


def run():
    text = prepare_text(open("Texts/OFilhoEterno-CristovaoTezza.txt", "r", encoding="utf8").read())
    top_words = get_most_frequent_words(text, 10)
    tokens = text_to_tokens(text)

    # Model made from the most frequent words
    model = Word2Vec(tokens, vector_size=50, sg=1, min_count=min(top_words, key=lambda t: t[1])[1])
    plot_pca(model)

    # Reducing the word array using the minimum count of 30 to improve the plot readability
    model = Word2Vec(tokens, vector_size=50, sg=1, min_count=30)
    plot_pca(model)

    # Using the default minimum count (5) but with no word annotation to improve the plot readability
    model = Word2Vec(tokens, vector_size=50, sg=1)
    plot_pca(model, False)

    show_similarities(model, top_words)


if __name__ == '__main__':
    run()
