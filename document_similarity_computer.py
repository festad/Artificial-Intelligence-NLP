import json
from typing import Dict, List

import calculator
from encoder import _get_tfidf_vector, get_term_frequency_vector, get_hot_encoded_vector
from navigator import retrieve_all_relevant_reuters
from preprocesser import preprocessing
from reuter_handler import _get_content, _get_id


def compute_document_cosine_similarity(text1: str, text2: str, encoding_type: str, algorithm: str, dictionary: Dict[str, int],
                                       idf: Dict[str, float]) -> float:
    if encoding_type == 'hot_encoding':
        hot_encoded_txt1 = get_hot_encoded_vector(preprocessing(text1), dictionary)
        hot_encoded_txt2 = get_hot_encoded_vector(preprocessing(text2), dictionary)
        if algorithm == 'cosine_distance':
            return calculator.cosine_similarity(hot_encoded_txt1, hot_encoded_txt2)
    elif encoding_type == 'term_frequency':
        term_frequency_encoded_txt1 = get_term_frequency_vector(preprocessing(text1), dictionary)
        term_frequency_encoded_txt2 = get_term_frequency_vector(preprocessing(text2), dictionary)
        if algorithm == 'cosine_distance':
            return calculator.cosine_similarity(term_frequency_encoded_txt1, term_frequency_encoded_txt2)
    elif encoding_type == 'tfidf':
        tfidf_encoding_txt1 = _get_tfidf_vector(preprocessing(text1), idf)
        tfidf_encoding_txt2 = _get_tfidf_vector(preprocessing(text2), idf)
        if algorithm == 'cosine_distance':
            return calculator.cosine_similarity(tfidf_encoding_txt1, tfidf_encoding_txt2)


def compute_document_cosine_similarity_from_local_encodings(query_words: List[str], newid: str,
                                                            encoding_type: str) -> float:
    if encoding_type == 'hot_encoding':
        with open(f'encoded/one_hot/{newid}.json', 'r') as f:
            tmp_document = json.load(f)
    elif encoding_type == 'term_frequency':
        with open(f'encoded/term_frequency/{newid}.json', 'r') as f:
            tmp_document = json.load(f)
    elif encoding_type == 'tfidf':
        with open(f'encoded/tf_idf/{newid}.json', 'r') as f:
            tmp_document = json.load(f)
    return calculator.cosine_similarity(query_words, tmp_document)


def document_query(qry: str, encoding_type: str) -> Dict[str, float]:
    with open('dictionary.json', 'r') as infile:
        d = json.load(infile)
    with open('idf.json', 'r') as infile:
        idf = json.load(infile)
    result = {}
    counter = 0
    preprocessed = preprocessing(qry)
    if encoding_type == 'hot_encoding':
        query_vector = get_hot_encoded_vector(preprocessed, d)
    elif encoding_type == 'term_frequency':
        query_vector = get_term_frequency_vector(preprocessed, d)
    elif encoding_type == 'tfidf':
        query_vector = _get_tfidf_vector(preprocessed, idf)
    for reut in retrieve_all_relevant_reuters():
        # print(_get_content(reut))
        result[_get_id(reut)] = compute_document_cosine_similarity_from_local_encodings(query_vector, _get_id(reut),
                                                                                        encoding_type)
        # compute_document_similarity(qry, _get_content(reut), encoding_type, algorithm_type, d, idf)
        counter += 1
        if counter % 1000 == 0:
            print(counter)
    return {k: v for k, v in
            sorted(result.items(),
                   key=lambda item: item[1], reverse=True)}
