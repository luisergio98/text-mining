import nltk
import pandas as pd
import numpy as np
import regex as re
import collections
import random
from matplotlib import pyplot as plt
from gensim.models import Word2Vec
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


def prepare_text(s):
    text = s.replace("\n", " ").replace("-", " ").lower()
    text = re.sub('[–?!@#$;—:,*~()‘”“]', '', text)
    pt_stp_words = stopwords.words('portuguese')
    word_array = [word for word in text.split() if word not in pt_stp_words]
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
    while count > 0:
        word_1 = word_2 = ' '
        while len([word for word in word_array if word_1 in word and word_2 in word]) > 0 or word_1 == word_2:
            word_1 = top_words[random.randint(0, len(top_words) - 1)][0]
            word_2 = top_words[random.randint(0, len(top_words) - 1)][0]

        print("The similarity between '" + word_1 + "' and '" + word_2 + "' is: ", end="")
        print(model.wv.similarity(word_1, word_2) * 100, end="%\n")
        word_array.append((word_1, word_2))
        count -= 1


def plot_pca(model, content=None, top=False):
    if top:
        x = [model.wv[word] for word in model.wv.key_to_index if len([top for top in content if word in top]) > 0]
        all_words = [word[0] for word in content]
    else:
        x = model.wv[model.wv.key_to_index]
        all_words = list(model.wv.key_to_index)

    x_corr = pd.DataFrame(x).corr()
    values, vectors = np.linalg.eig(x_corr)

    args = (-values).argsort()
    vectors = vectors[:, args]
    new_vectors = vectors[:, :2]

    new_x = np.dot(x, new_vectors)

    plt.figure(figsize=(13, 7))
    plt.scatter(new_x[:, 0], new_x[:, 1], linewidths=10, color='green')
    plt.xlabel("PC1", size=15)
    plt.ylabel("PC2", size=15)
    plt.title("Word Embedding Space", size=20)
    for i, word in enumerate(all_words):
        plt.annotate(word, xy=(new_x[i, 0], new_x[i, 1]))

    plt.show()


def run():
    text = prepare_text(open("Texts/OFilhoEterno-CristovaoTezza.txt", "r", encoding="utf8").read())

    top_words = get_most_frequent_words(text, 10)

    tokens = text_to_tokens(text)

    model = Word2Vec(tokens, vector_size=50, sg=1, min_count=30)

    show_similarities(model, top_words)

    plot_pca(model)


if __name__ == '__main__':
    run()
