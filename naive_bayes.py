'''
The function in this file assumes that you
already have the files:
    topic_inverted_index.json
    topic_nonverted_index.json
    topics_frequencies.json
    dictionary.json
If you don't have them, use immediately
    'indexer.py'
with which you can create these and other
important files for other classifiers.
'''

import json
from math import log
from typing import Dict

from preprocesser import preprocessing


def naive_bayes(query: str) -> Dict[str, float]:
    '''
    'topic_inverted_index' stores a dictionary
    whose key is a word and the value is another
    dictionary whose key is a topic and the value
    is the number of times the word appears in that topic.
    'topic_nonverted_index' is similar, it is a dictionary
    whose key is a topic and the value is another dictionary
    whose key is a word and the value is the number of times
    the word appears in the document.
    It is useful to have both of them because
    the nested for loop are easier to write then
    if only one of them, in fact the first one contains
    all the words ever found in the collection, whereas
    the second one shows for each topic only the words that
    appear in that topic, but his advantage is that it contains
    all the topics in the set of keys
    Naive Bayes maximizes:
    P(c)*P(d|c),
    P(c) is in the 'topic_frequencies.json' file,
    P(d|c) is the product of all P(t_i|c),
    and the probability that a word appears in a given class
    is the number of times it appears in the class
        (here 'topic_inverted_index' is useful,
        it can be checked fast it a word appears in a topic or not,
        to decide if the result is 0 or another integer)
    divided by the sum of the times
    every word appears in that class
        (here 'topic_nonverted_index' is useful, for a given
        topic it's easy to sum all the values of all the words in that topic)

    'topic_scores' is a dictionary that for every topic
    stores the probability of that topic to match the query

    Instead of calculating products of probabilities,
    sums of logarithm of probabilities are computed, otherwise
    for too long queries the result would become too small values
    that the machine would treat as 0.
    '''
    with open('topic_inverted_index.json', 'r') as f:
        topic_inverted_index = json.load(f)
    with open('topic_nonverted_index.json', 'r') as f:
        topic_nonverted_index = json.load(f)
    with open('topics_frequencies.json') as f:
        topics_frequencies = json.load(f)
    with open('dictionary.json', 'r') as f:
        dictionary = json.load(f)
    smoother = len(dictionary)
    topics_scores = {}
    for t in topics_frequencies:
        topics_scores[t] = log(topics_frequencies[t])
    preprocessed_query = preprocessing(query)
    for t in topics_frequencies:
        for w in preprocessed_query:
            tmp_score_numerator = 0
            if w in topic_inverted_index:
                tmp_score_numerator = topic_inverted_index[w][t] if \
                    t in topic_inverted_index[w] else 0
            topics_scores[t] += \
                log((1 + tmp_score_numerator) / \
                (sum(topic_nonverted_index[t].values()) + smoother))
    return {k: v for k,v in sorted(topics_scores.items(), key=lambda item: item[1], reverse=True)}