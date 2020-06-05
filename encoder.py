'''
The function in this file assumes that you
already have the files:
    dictionary.json
    idf.json
for cosine distance,
and also
    topics_dictionary.json
to calculate the target vectors (used to compute
the error of the result from the last layer
of the neural network), vectors that have the same dimension
as the number of the different topics and are made of 1 (when
the corresponding topic belongs to that particular document) or 0.
If you don't have them, use immediately
    'indexer.py'
with which you can create these and other
important files for other classifiers.
'''

import json
from math import log
from typing import List, Dict

from navigator import retrieve_all_relevant_reuters
from preprocesser import preprocessing
from reuter_handler import _preprocess_content, _get_id, _get_topics


def get_hot_encoded_vector(words: List[str], dictionary: Dict[str, int]) -> List[int]:
    array_dictionary = [k for k in dictionary]
    vector = [0 for _ in range(len(array_dictionary))]
    for w in words:
        if w in array_dictionary:
            vector[array_dictionary.index(w)] = 1
    # print(vector)
    return vector


def get_term_frequency_vector(words: List[str], dictionary: Dict[str, int]) -> List[int]:
    array_dictionary = [k for k in dictionary]
    vector = [0 for _ in range(len(array_dictionary))]
    map_wds = {}
    for w in words:
        if w in map_wds:
            map_wds[w] += 1
        else:
            map_wds[w] = 1
    for w in words:
        if w in array_dictionary:
            vector[array_dictionary.index(w)] = map_wds[w]
    return vector


def _get_tfidf_vector(words: List[str], idf: Dict[str, float]) -> List[float]:
    array_dictionary = [k for k in idf]
    vector = [0 for _ in range(len(array_dictionary))]
    map_wds = {}
    for w in words:
        if w in map_wds:
            map_wds[w] += 1
        else:
            map_wds[w] = 1
    for w in words:
        if w in array_dictionary:
            # print(f'{w} -> {idf[w]}')
            vector[array_dictionary.index(w)] = log(1 + map_wds[w]) * idf[w]
    return vector


def get_tf_idf_vector(text: str) -> List[float]:
    qry = preprocessing(text)
    with open('idf.json', 'r') as f:
        idf = json.load(f)
    return _get_tfidf_vector(qry, idf)


def load_tfidf_vector(tfidf, reut):
    words = _preprocess_content(reut)
    array_dictionary = [k for k in tfidf]
    vector = [0 for _ in range(len(array_dictionary))]
    for w in words:
        if w in array_dictionary:
            vector[array_dictionary.index(w)] = tfidf[w][_get_id(reut)]
    return vector


def write_encodings():
    '''
    Make sure to have the directory
        encoded/one_hot
        encoded/term_frequency
        encoded/tf_idf
    '''
    with open('dictionary.json', 'r') as f:
        dictionary = json.load(f)
    with open('idf.json', 'r') as f:
        idf = json.load(f)
    i = 1
    for reuter in retrieve_all_relevant_reuters():
        newid = _get_id(reuter)
        preprocessed = _preprocess_content(reuter)
        hot_encoded = get_hot_encoded_vector(preprocessed, dictionary)
        with open(f'encoded/one_hot/{newid}.json', 'w') as f:
            json.dump(hot_encoded, f)
        term_frequency = get_term_frequency_vector(preprocessed, dictionary)
        with open(f'encoded/term_frequency/{newid}.json', 'w') as f:
            json.dump(term_frequency, f)
        tfidf = _get_tfidf_vector(preprocessed, idf)
        with open(f'encoded/tf_idf/{newid}.json', 'w') as f:
            json.dump(tfidf, f)
        print(i)
        i += 1


def write_targets():
    with open('topics_dictionary.json', 'r') as f:
        topics_dictionary = json.load(f)
        k = 1
    for reuter in retrieve_all_relevant_reuters():
        newid = _get_id(reuter)
        topics = _get_topics(reuter)
        for t in topics_dictionary:
            if t in topics:
                topics_dictionary[t] = 1
            else:
                topics_dictionary[t] = 0
        vector = []
        for t in topics_dictionary:
            vector.append(topics_dictionary[t])
        with open(f'training_nn/{newid}.json', 'w') as f:
            json.dump(vector, f)
        print(k)
        k += 1
