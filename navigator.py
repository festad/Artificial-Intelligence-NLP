'''
This one manages the communication with
reuters that are written in in sgm format,

'''

import os
from typing import List

from bs4 import BeautifulSoup


def _file_names_in_dir(directory='reut') -> List[str]:
    return [f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))]


def _generator_of_file_names_in_dir(directory='reut'):
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            yield f


def _filter_reuts_names(files: List[str]) -> List[str]:
    return [f for f in files if f.startswith('reut')]


def retrieve_reuts_names(directory='reut') -> List[str]:
    return _filter_reuts_names(
        _file_names_in_dir(directory))


def _file_content(filename: str, directory='reut') -> str:
    with open(f'{directory}/{filename}', "r") as fi:
        return fi.read()


def _get_separate_reuters(filename: str, directory='reut'):
    text = _file_content(filename, directory)
    soup = BeautifulSoup(text, "lxml")
    return soup.find_all("reuters")


def _filter_train(reuters):
    return [reut for reut in reuters if reut['lewissplit'] == 'TRAIN']


def _filter_relevant_reuters_for_train(reuters):
    return [reut for reut in reuters if reut["lewissplit"] == "TRAIN"
            and reut["topics"] == "YES"
            and reut.content is not None
            and len(reut.topics.find_all("d")) >= 1]


def retrieve_all_reuters():
    files = retrieve_reuts_names()
    for file in files:
        reuters = _get_separate_reuters(file)
        for reut in reuters:
            yield reut


def retrieve_all_relevant_reuters():
    '''
    Relevant reuters are the ones with
    'TRAIN' tag and at least one topic
    '''
    files = retrieve_reuts_names()
    for file in files:
        reuters = _filter_relevant_reuters_for_train(
            _get_separate_reuters(file))
        for reut in reuters:
            yield reut


def count_all_relevant_reuters():
    iterator = retrieve_all_relevant_reuters()
    i = 0
    for reut in iterator:
        i += 1
    return i