'''
This one manages Rocchio classifier.
'''


import json
from typing import List, Dict

from calculator import centroid, cosine_similarity
from encoder import get_hot_encoded_vector, get_term_frequency_vector, _get_tfidf_vector
from navigator import _generator_of_file_names_in_dir
from preprocesser import preprocessing


def write_encoded_vectors_for_any_topic(encoding_type: str):
    with open('document_topic_dictionary.json', 'r') as f:
        document_topic_dictionary = json.load(f)
    topic_vectors = {}
    j = 1
    if encoding_type == 'hot_encoding':
        for file in _generator_of_file_names_in_dir('encoded/one_hot'):
            with open(f'encoded/one_hot/{file}', 'r') as f:
                hot_encoding_vector = json.load(f)
            newid = file.split('.json')[0]
            this_topics = document_topic_dictionary[newid]
            for t in this_topics:
                if t not in topic_vectors:
                    topic_vectors[t] = []
                topic_vectors[t].append(hot_encoding_vector)
            print(j)
            j += 1
        with open('topic_hot_encoded_vectors.json', 'w') as f:
            json.dump(topic_vectors, f, sort_keys=True)
        return topic_vectors

    elif encoding_type == 'term_frequency':
        for file in _generator_of_file_names_in_dir('encoded/term_frequency'):
            with open(f'encoded/term_frequency/{file}', 'r') as f:
                term_frequency_vector = json.load(f)
            newid = file.split('.json')[0]
            this_topics = document_topic_dictionary[newid]
            for t in this_topics:
                if t not in topic_vectors:
                    topic_vectors[t] = []
                topic_vectors[t].append(term_frequency_vector)
            print(j)
            j += 1
        with open('topic_term_frequency_vectors.json', 'w') as f:
            json.dump(topic_vectors, f, sort_keys=True)
        return

    elif encoding_type == 'tfidf':
        for file in _generator_of_file_names_in_dir('encoded/tf_idf'):
            with open(f'encoded/tf_idf/{file}', 'r') as f:
                tfidf_vector = json.load(f)
            newid = file.split('.json')[0]
            this_topics = document_topic_dictionary[newid]
            for t in this_topics:
                if t not in topic_vectors:
                    topic_vectors[t] = []
                topic_vectors[t].append(tfidf_vector)
            print(j)
            j += 1
        with open('topic_tfidf_vectors.json', 'w') as f:
            json.dump(topic_vectors, f, sort_keys=True)
        return


def compute_centroids(encoding_type: str):
    '''
    Depending on which of the three different encodings,
    a different topic_centroid is computed.
    'topic_centroid' is a dictionary that for each different
    topic stores a vector, that is the average of all the
    vectors (derived from documents) in that topic.
    '''
    if encoding_type == 'hot_encoding':
        with open('topic_hot_encoded_vectors.json', 'r') as f:
            topic_hot_encoded_vector = json.load(f)
        topic_centroid = {}
        for t in topic_hot_encoded_vector:
            topic_centroid[t] = centroid(topic_hot_encoded_vector[t])
        with open('topic_hot_encoded_centroid.json', 'w') as f:
            json.dump(topic_centroid, f)
    elif encoding_type == 'term_frequency':
        with open('topic_term_frequency_vectors.json', 'r') as f:
            topic_term_frequency_vector = json.load(f)
        topic_centroid = {}
        for t in topic_term_frequency_vector:
            topic_centroid[t] = centroid(topic_term_frequency_vector[t])
        with open('topic_term_frequency_centroid.json', 'w') as f:
            json.dump(topic_centroid, f)
    elif encoding_type == 'tfidf':
        with open('topic_tfidf_vectors.json', 'r') as f:
            topic_tfidf_vector = json.load(f)
        topic_centroid = {}
        for t in topic_tfidf_vector:
            topic_centroid[t] = centroid(topic_tfidf_vector[t])
        with open('topic_tfidf_centroid.json', 'w') as f:
            json.dump(topic_centroid, f)


def rocchio(query: str, encoding_type: str) -> Dict[str, float]:
    '''
    Depending on which of the three encoding types,
    given a query, the corresponding vector is computed
    and then for each of the different centroids related
    to a particular topic, the cosine similarity between
    the query vector and the centroid,
    then a sorted dictionary is returned, whose key is the topic
    and the value is the similarity with the query vector
    '''
    with open('dictionary.json', 'r') as f:
        dictionary = json.load(f)
    with open('idf.json', 'r') as f:
        idf = json.load(f)
    if encoding_type == 'hot_encoding':
        hot_encoded_qry = get_hot_encoded_vector(preprocessing(query), dictionary)
        topic_scores = {}
        with open('topic_hot_encoded_centroid.json', 'r') as f:
            hot_encoded_centroid = json.load(f)
        for t in hot_encoded_centroid:
            topic_scores[t] = cosine_similarity(hot_encoded_centroid[t], hot_encoded_qry)
        return {k: v for k, v in sorted(topic_scores.items(), key=lambda item: item[1], reverse=True)}
    elif encoding_type == 'term_frequency':
        term_frequency_qry = get_term_frequency_vector(preprocessing(query), dictionary)
        topic_scores = {}
        with open('topic_term_frequency_centroid.json', 'r') as f:
            term_frequency_centroid = json.load(f)
        for t in term_frequency_centroid:
            topic_scores[t] = cosine_similarity(term_frequency_centroid[t], term_frequency_qry)
        return {k: v for k, v in sorted(topic_scores.items(), key=lambda item: item[1], reverse=True)}
    elif encoding_type == 'tfidf':
        tfidf_query = _get_tfidf_vector(preprocessing(query), idf)
        topic_scores = {}
        with open('topic_term_frequency_centroid.json', 'r') as f:
            tfidf_centroid = json.load(f)
        for t in tfidf_centroid:
            topic_scores[t] = cosine_similarity(tfidf_centroid[t], tfidf_query)
        return {k: v for k, v in sorted(topic_scores.items(), key=lambda item: item[1], reverse=True)}