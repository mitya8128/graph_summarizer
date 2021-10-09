from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters
from string import punctuation

need_tag = True

class Configuration():
    #TODO: think about decent approach to pass need_tag variable
    MORPH = MorphAnalyzer()
    if need_tag:
        STOP_WORDS = stopwords.words('russian')
    else:
        STOP_WORDS_EN = stopwords.words('english')
    PUNCT = punctuation + '«»—…“”*№–'
    TO_UD = converters.converter('opencorpora-int', 'ud20')

    MODEL_PATH = 'models/model.bin'
    MODEL = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)


conf = Configuration()