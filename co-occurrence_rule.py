import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from apyori import apriori
from operator import itemgetter
from wordcloud import WordCloud
from nltk.corpus import stopwords


def prepare_text(s):
    nltk.download('stopwords')
    t = s.replace("\n", " ").replace("-", " ").lower()
    t = re.sub('[–?!@#$;—:,*~()‘”“]', '', t)
    word_array = [word for word in t.split() if word not in stopwords.words('portuguese')]
    return " ".join(word_array)


def show_co_occurrence(t):
    phrases = t.split(sep='.')
    dict_list = [{}, {}, {}, {}]

    for i in range(0, len(phrases)):
        words = re.split(' ', phrases[i])
        new = []
        for w in words:
            lower_w = w.lower().strip()
            if len(w) < 2:
                continue
            if lower_w in stopwords.words('portuguese'):
                continue
            if lower_w in dict_list[0]:
                dict_list[0][lower_w] = dict_list[0][lower_w] + 1
            else:
                dict_list[0][lower_w] = 1
            new.append(lower_w)

        if len(new) > 1:
            for j in range(0, len(new) - 1):
                two = str(new[j]) + ' ' + str(new[j + 1])
                if two in dict_list[1]:
                    dict_list[1][two] = dict_list[1][two] + 1
                else:
                    dict_list[1][two] = 1
        if len(new) > 2:
            for j in range(0, len(new) - 2):
                three = str(new[j]) + ' ' + str(new[j + 1]) + ' ' + str(new[j + 2])
                if three in dict_list[2]:
                    dict_list[2][three] = dict_list[2][three] + 1
                else:
                    dict_list[2][three] = 1
        if len(new) > 2:
            for j in range(0, len(new) - 3):
                quad = str(new[j]) + ' ' + str(new[j + 1]) + ' ' + str(new[j + 2]) + ' ' + str(new[j + 3])
                if quad in dict_list[3]:
                    dict_list[3][quad] = dict_list[3][quad] + 1
                else:
                    dict_list[3][quad] = 1

    for i, d in enumerate(dict_list):
        print("\n" + str(i + 1) + " Word(s): ")
        print(sorted(d.items(), key=itemgetter(1), reverse=True)[:10], end="\n")


def show_rule_model(t):
    phrases = t.split(sep='.')
    for i in range(0, len(phrases)):
        phrases[i] = re.sub(r'==.*?==+', '', phrases[i])

    new_phrases = list()
    for i in range(0, len(phrases)):
        words = re.split(' ', phrases[i])
        new = []
        for w in words:
            lower_w = w.lower().strip()
            if len(w) < 2:
                continue
            if lower_w in stopwords.words('portuguese'):
                continue
            new.append(lower_w)
        if len(new) > 1:
            new_phrases.append(new)

    rules = apriori(new_phrases, min_support=0.004, min_confidence=0.2, min_lift=3, min_length=2)

    r = pd.DataFrame(list(rules))
    print("\n")
    print(r.head(20))
    print(r.tail(20))


def show_word_cloud(t):
    word_cloud = WordCloud(stopwords=stopwords.words('portuguese'),
                           background_color="black",
                           width=1600, height=800).generate(t)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(word_cloud, interpolation="bilinear")
    ax.set_axis_off()

    plt.imshow(word_cloud)
    plt.show()


def run():
    t = prepare_text(open("Texts/OFilhoEterno-CristovaoTezza.txt", "r", encoding="utf8").read())

    show_co_occurrence(t)

    show_rule_model(t)

    show_word_cloud(t)


if __name__ == '__main__':
    run()
