import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import scipy
import re
from graph_utils import find_cliques_all
from utils import text_cleaner
from wordrank import wordrank

stop_words = stopwords.words('russian')


# funcs for TextRank

def read_article(file_name):
    with open(file_name, 'rb') as f:
        text = f.readlines()
    text = text[0].decode("mac_cyrillic")
    sentences = text[0].split("\r")
    for sentence in sentences:
        print(sentence)
    # return sentences


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(sentences: list) -> list:
    sentences_cleaned = []
    for i in range(len(sentences)):
        sentence_clean = text_cleaner(sentences[i], stop_words)
        sentences_cleaned.append((sentence_clean, i))

    sentences_cleaned_fix = [elem for elem in sentences_cleaned if elem[0] != '']

    ranked_list = []
    for i in range(len(sentences_cleaned_fix)):
        sent_ranked = wordrank(sentences_cleaned_fix[i][0])
        ranked_list.append((sent_ranked, sentences_cleaned_fix[i][1]))

    ranked_list_notags = []

    for i in range(len(ranked_list)):
        sent_without_tags = ' '.join([" ".join(word.split("_")[0] for word in s.split()) for s in ranked_list[i][0]])
        ranked_list_notags.append((sent_without_tags, ranked_list[i][1]))

    ranked_list_notags_nonums = []
    for i in range(len(ranked_list_notags)):
        element = ranked_list_notags[i][0]
        ranked_list_notags_nonums.append(element)

    sentence_similarity_matrix = build_similarity_matrix(ranked_list_notags_nonums, stop_words)

    # Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    cliques = find_cliques_all(sentence_similarity_graph)
    min_clique = cliques[0]

    snts = []    # но как мы восстановим оригинальные индексы? --> наверное сравнением элементов списка
    for i in min_clique:
        sent = sentences[i]
        snts.append(sent)

    return snts


def generate_summary_loop(text, n, n_compression=True):

    if n_compression:
        abstract = generate_summary(text)
        while n > 0:
            n -= 1
            abstract = generate_summary(abstract)
    else:
        abstract = generate_summary_debug(text)

    return abstract


