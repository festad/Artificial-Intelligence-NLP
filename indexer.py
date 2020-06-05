'''
This file is of extreme importance,
it allows you to create the files used as resources
from the classifiers:
    naive bayes,
    optimized cosine distance.
The suggestion is to open up a python console,
import this file and call the functions
that start with 'write' manually.
The bad news is that some of this functions
write files using others,
the good news is that if you follow
the order in which the functions are written in this file
you won't have problem, and even if you had, you
would get an error message like
'FileNotFoundError' and you'll easily
figure out what is the dependency you are missing.
Attention:
this file provides NO service for
    Rocchio
    Cosine distance (that can use
        one hot / bag of words encoding
        term frequency         encoding
        tfidf                  encoding),
    Neural network
    Rocchio
if you want to use Rocchio's classifier,
the needed files are not created by function inside
this file, you'll have to look inside 'rocchio.py';
the same goes for neural newtork classifier.
It seems strange that this file provide no service
for Cosine distance because actually some functions
that write idf and tfidf encodings, that's because
they will be used (only) by optimized cosine distance classifier.
The files to be used for cosine distance classifier are
    'document_similarity_computer.py'
    'encoder.py',
the first one used the vectors written by the second one.
'''


import json
from math import log
from typing import List, Dict

from navigator import retrieve_all_relevant_reuters, count_all_relevant_reuters
from reuter_handler import _preprocess_content, _get_id, _get_topics


def _update_inverted_index(docID: str, words: List[str],
                           inverted_index: Dict[str, Dict[str, int]]):
    for w in words:
        if w in inverted_index:
            if docID in inverted_index[w]:
                inverted_index[w][docID] += 1
            else:
                inverted_index[w][docID] = 1
        else:
            inverted_index[w] = {}
            inverted_index[w][docID] = 1


def write_inverted_index():
    inverted_index = {}
    counter = 1
    for reuter in retrieve_all_relevant_reuters():
        _update_inverted_index(_get_id(reuter),
                               _preprocess_content(reuter), inverted_index)
        print(counter)
        counter += 1
    with open('inverted_index.json', 'w') as f:
        json.dump(inverted_index, f, indent=4, sort_keys=True)


def _update_dictionary(words: List[str], dictionary: Dict[str, int]):
    for w in words:
        if w in dictionary:
            dictionary[w] += 1
        else:
            dictionary[w] = 1


def write_dictionary():
    dictionary = {}
    counter = 1
    for reuter in retrieve_all_relevant_reuters():
        _update_dictionary(_preprocess_content(reuter), dictionary)
        print(counter)
        counter += 1
    with open('dictionary.json', 'w') as f:
        json.dump(dictionary, f, indent=4, sort_keys=True)


def _update_topic_inverted_index(topics: List[str], words: List[str], topic_inverted_index: Dict[str, Dict[str, int]]):
    for w in words:
        if w in topic_inverted_index:
            for topic in topics:
                if topic in topic_inverted_index[w]:
                    topic_inverted_index[w][topic] += 1
                else:
                    topic_inverted_index[w][topic] = 1
        else:
            for topic in topics:
                topic_inverted_index[w] = {}
                topic_inverted_index[w][topic] = 1


def write_topic_inverted_index():
    topic_inverted_index = {}
    counter = 1
    for reuter in retrieve_all_relevant_reuters():
        _update_topic_inverted_index(_get_topics(reuter),
                                     _preprocess_content(reuter), topic_inverted_index)
        print(counter)
        counter += 1
    with open('topic_inverted_index.json', 'w') as f:
        json.dump(topic_inverted_index, f, indent=4, sort_keys=True)


def _update_topic_nonverted_index(topics: List[str], words: List[str],
                                  topic_nonverted_index: Dict[str, Dict[str, int]]):
    for topic in topics:
        if topic not in topic_nonverted_index:
            topic_nonverted_index[topic] = {}
        for w in words:
            if w in topic_nonverted_index[topic]:
                topic_nonverted_index[topic][w] += 1
            else:
                topic_nonverted_index[topic][w] = 1


def write_topic_nonverted_index():
    topic_nonverted_index = {}
    counter = 1
    for reuter in retrieve_all_relevant_reuters():
        _update_topic_nonverted_index(_get_topics(reuter),
                                      _preprocess_content(reuter), topic_nonverted_index)
        print(counter)
        counter += 1
    with open('topic_nonverted_index.json', 'w') as f:
        json.dump(topic_nonverted_index, f, indent=4, sort_keys=True)


def _update_topics_dictionary(topics: List[str], topics_dictionary: Dict[str, int]):
    for topic in topics:
        if topic in topics_dictionary:
            topics_dictionary[topic] += 1
        else:
            topics_dictionary[topic] = 1


def write_topics_dictionary():
    topics_dictionary = {}
    counter = 1
    for reuter in retrieve_all_relevant_reuters():
        _update_topics_dictionary(_get_topics(reuter), topics_dictionary)
        print(counter)
        counter += 1
    with open('topics_dictionary.json', 'w') as f:
        json.dump(topics_dictionary, f, indent=4, sort_keys=True)


def write_topics_frequencies():
    topics_frequencies = {}
    with open('topics_dictionary.json', 'r') as f:
        topics_dictionary = json.load(f)
    total = sum(topics_dictionary.values())
    i = 1
    for t in topics_dictionary:
        topics_frequencies[t] = topics_dictionary[t] / total
        print(i)
        i += 1
    with open('topics_frequencies.json', 'w') as f:
        json.dump(topics_frequencies, f, indent=4, sort_keys=True)


def _generate_idf(inverted_index: Dict[str, Dict[str, int]], number_documents: int) -> Dict[str, float]:
    idf = {}
    i = 1
    for w in inverted_index:
        idf[w] = log(number_documents / len(inverted_index[w]))
        print(i)
        i += 1
    return idf


def write_idf():
    with open('inverted_index.json', 'r') as f:
        inverted_index = json.load(f)
    number_documents = count_all_relevant_reuters()
    idf = _generate_idf(inverted_index, number_documents)
    with open('idf.json', 'w') as f:
        json.dump(idf, f)


def _generate_tfidf(inverted_index: Dict[str, Dict[str, int]], idf: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    tfidf = {}
    i = 0
    for w in inverted_index:
        i += 1
        tfidf[w] = {}
        for d in inverted_index[w]:
            tfidf[w][d] = log(1 + inverted_index[w][d]) * idf[w]
        print(i)
    return tfidf


def write_tfidf():
    with open('inverted_index.json', 'r') as f:
        inverted_index = json.load(f)
    with open('idf.json', 'r') as f:
        idf = json.load(f)
    tfidf = _generate_tfidf(inverted_index, idf)
    with open('tfidf.json', 'w') as f:
        json.dump(tfidf, f)


def _generate_inverse_tfidf(inverted_index: Dict[str, Dict[str, int]], idf: Dict[str, float]) \
        -> Dict[str, Dict[str, float]]:
    inv_tfidf = {}
    i = 1
    for w in inverted_index:
        print(i)
        i += 1
        print(w)
        for doc in inverted_index[w]:
            inv_tfidf[doc] = {}
    i = 1
    for w in inverted_index:
        print(i)
        i += 1
        print(w)
        for doc in inverted_index[w]:
            inv_tfidf[doc][w] = inverted_index[w][doc]
    i = 1
    for doc in inv_tfidf:
        print(i)
        i += 1
        print(doc)
        for w in inv_tfidf[doc]:
            inv_tfidf[doc][w] = log(1 + inv_tfidf[doc][w]) * idf[w]
    return inv_tfidf


def write_inverse_tfidf():
    with open('inverted_index.json', 'r') as f:
        inverted_index = json.load(f)
    with open('idf.json', 'r') as f:
        idf = json.load(f)
    inv_tfidf = _generate_inverse_tfidf(inverted_index, idf)
    with open('inv_tfidf.json', 'w') as f:
        json.dump(inv_tfidf, f)


def write_documents_topics():
    document_topic_dictionary = {}
    j = 1
    for r in retrieve_all_relevant_reuters():
        newid = _get_id(r)
        topics = _get_topics(r)
        document_topic_dictionary[newid] = []
        for t in topics:
            document_topic_dictionary[newid].append(t)
        print(j)
        j += 1
    with open('document_topic_dictionary.json', 'w') as f:
        json.dump(document_topic_dictionary, f, indent=4, sort_keys=True)
