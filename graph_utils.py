import networkx as nx
import numpy as np
from utils import cosine, pymorphy_tagger, clean_numbers, vectorize_word, getList
from config import conf
import operator


stop_words = conf.STOP_WORDS
model = conf.MODEL


def similar_words(text, n):
    """return n most similar words in models dictionary"""
    try:
        lst = model.most_similar(text, topn=n)
    except KeyError:
        lst = []

    return lst


def vertices(text, n):
    """return list of vertices based on similar_words function"""

    vertices = similar_words(text, n)
    vertices_list = [vertices[0] for vertices in vertices]

    return vertices_list


def adjacency_mat(vertices_list):
    """make matrix of distances between words"""
    n = len(vertices_list)
    adj_mat = []
    for i in vertices_list:
        for j in vertices_list:
            adj_vec = []
            vec = cosine(vectorize_word(i), vectorize_word(j))
            adj_vec.append(vec)
            adj_mat.append(adj_vec)

    return np.array(adj_mat).reshape(n, n)


def make_graph(mat, vertices_list, th):
    """make graph with edges between vertices based on adjacency_mat function"""

    G = nx.from_numpy_matrix(mat)
    mapping = dict(zip(G, vertices_list))
    H = nx.relabel_nodes(G, mapping)
    labels = nx.get_edge_attributes(H, 'weight')
    labels_filtered = dict()

    for (key, value) in labels.items():
        if value <= th:
            labels_filtered[key] = value
        else:
            pass

    e = getList(labels_filtered)

    for element in e:
        H.remove_edge(*element)

    return H


def text2graph(text, th, raw=True):
    """full pipeline to make graph from text"""
    # if raw:
    #     text_tagged = pymorphy_tagger(clean_numbers(text), stop_words)
    #     text_tagged = text_tagged.split()
    # else:
    #     text_tagged = list(text)
    vertices_list = vertices(text, 5)
    text_mat = adjacency_mat(vertices_list)
    graph = make_graph(text_mat, vertices_list, th)

    return graph, text_mat


def find_maxlen_clique(grph) -> list:
    """find clique with max len"""

    # list_cliques = list(find_cliques(grph))
    list_cliques = find_cliques_all(grph)
    list_cliques_len = []
    for i in range(len(list_cliques)):
        len_elmnt = len(list_cliques[i])
        list_cliques_len.append(len_elmnt)
    index, value = max(enumerate(list_cliques_len), key=operator.itemgetter(1))
    return list_cliques[index]


def find_cliques_all(graph) -> list:
    """simple interface for nx function
    returns list of all cliques"""
    return list(nx.algorithms.clique.find_cliques(graph))
