from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters
from string import punctuation


class Configuration:
    MORPH = MorphAnalyzer()
    STOP_WORDS = stopwords.words('russian')
    PUNCT = punctuation + '«»—…“”*№–'
    TO_UD = converters.converter('opencorpora-int', 'ud20')

    MODEL_PATH = 'models/model.bin'
    MODEL = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)