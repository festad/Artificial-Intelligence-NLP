'''
The function in this file assumes that you
already have the files:
    idf.json
    inv_tfidf.json
If you don't have them, use immediately
    'indexer.py'
with which you can create these and other
important files for other classifiers.
Note that the creation of this two files
requires the previous creation of another,
called 'inverted_index.json'.
Also note that when this function
is used inside 'main.py'
it requires the file 'document_topic_dictionary.json'
which is not conceptually related to the
optimized cosine distance function
but it is needed to show which is the topic of
the document whose tfidf vector is closest to the query.
'''

import json
from math import sqrt, log
from typing import Dict

from preprocesser import preprocessing


def optimized_improved_cosine_distance_tfidf(qry: str) -> Dict[str, float]:
    '''
    It does the same thing that it is done in 'cosine_distance.py',
    but it does it faster, and only for tf_idf encoding.
    It is faster because instead of creating big arrays
    whose biggest part of elements is 0 (because the biggest part
    of all the word in the collection doesn't appear in the query or in a singular document),
    it creates small array containing only the needed values.
    The result is a sorted dictionary whose keys are the document ids and
    the values are the similarity between each document and the query
    'inv_tf_idf' is a dictionary whose key is a document id and the corresponding value
    is another dictionary whose key is a word and the value is
    the tf*idf value for that
    particular word in that document.
    '''
    with open('idf.json', 'r') as f:
        idf = json.load(f)
    with open('inv_tfidf.json', 'r') as f:
        inv_tfidf = json.load(f)
    words = preprocessing(qry)
    query_vector = {}
    for w in words:
        if w in idf:
            if w in query_vector:
                query_vector[w] += 1
            else:
                query_vector[w] = 1
    for w in query_vector:
        query_vector[w] = log(1 + query_vector[w]) * idf[w]
    doc_score = {}
    for doc in inv_tfidf:
        tmp = {}
        match_vector = {}
        for k in inv_tfidf[doc]:
            match_vector[k] = inv_tfidf[doc][k]
        for w in query_vector:
            if w not in match_vector:
                match_vector[w] = 0
            tmp[w] = query_vector[w] * match_vector[w]
        n = sum(tmp.values())
        if n != 0:
            mod_query_vector = sqrt(sum([n * n for n in query_vector.values()]))
            mod_match_vector = sqrt(sum([n * n for n in match_vector.values()]))
            d = mod_query_vector * mod_match_vector
            doc_score[doc] = n / d
        else:
            doc_score[doc] = 0
    return {k: v for k, v in sorted(doc_score.items(), key=lambda item: item[1], reverse=True)}
