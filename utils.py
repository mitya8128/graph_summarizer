import nltk
import numpy as np
import scipy
import random
import re

from config import conf
from russian_tagsets import converters
from numba import jit

morph = conf.MORPH
punct = conf.PUNCT
model = conf.MODEL

to_ud = converters.converter('opencorpora-int', 'ud20')


def clean_numbers(text):
    text = re.sub(r'[0-9]+', '', text)
    return text


def pymorphy_tagger(text, stops):
    text = text.replace('[', ' ').replace(']', ' ')
    parsed = []
    tokens = nltk.word_tokenize(text)
    for word in tokens:
        word = word.strip(punct)
        if (word not in stops) and (word not in punct) and (
                re.sub(r'[{}]+'.format(punct), '', word).isdigit() is False) and (word != 'nan'):
            lemma = str(morph.parse(word)[0].normal_form)
            pos = to_ud(str(morph.parse(word)[0].tag.POS)).split()[0]
            word_with_tag = lemma + '_' + pos
            parsed.append(word_with_tag)
    return ' '.join(parsed)


@jit
def cosine(a, b):
    dot = np.dot(a, b.T)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return cos


# wrapper to deal with numba's lack of support for try/except block (see https://github.com/numba/numba/issues/5377)
def cosine_wrapper(a , b):
    try:
        return cosine(a, b)
    except ZeroDivisionError:
        return 0


def cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def text_cleaner(text, stopwords):
    text = text.replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ')
    cleaned = []
    tokens = nltk.word_tokenize(text)
    for word in tokens:
        word = word.strip(punct)
        if (word not in stopwords) and (word not in punct) and (
                re.sub(r'[{}]+'.format(punct), '', word).isdigit() is False) and (word != 'nan'):
            lemma = str(morph.parse(word)[0].normal_form)
            cleaned.append(lemma)
    return ' '.join(cleaned)


def vectorize_word(word):
    """vectorize word with unknown word handler"""
    try:
        vec = model[word]
    except KeyError:
        vec = np.zeros(len(model[random.choice(model.index_to_key)]))

    return vec


def getList(dct):
    list = []
    for key in dct.keys():
        list.append(key)

    return list