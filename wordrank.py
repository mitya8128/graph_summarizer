import nltk
from nltk.corpus import stopwords
import numpy as np
import scipy
import re

from config import conf
from graph_utils import text2graph, find_cliques_all, adjacency_mat, make_graph, find_maxlen_clique
from utils import text_cleaner, pymorphy_tagger

model = conf.MODEL
stopwords = conf.STOP_WORDS


def wordrank(text: str, need_tag: bool) -> list:
    """returns clique with max len"""
    text_tagged = []
    text = text.split()

    if need_tag:
        for element in text:
            el = pymorphy_tagger(element, stopwords)
            text_tagged.append(el)
    else:
        text_tagged = text

    text_mat = adjacency_mat(text_tagged)
    graph = make_graph(text_mat, text_tagged, 0.3)
    maxlen_clique = find_maxlen_clique(graph)

    return maxlen_clique