import json
from typing import List, Set, Dict

from preprocesser import preprocessing
from navigator import retrieve_all_relevant_reuters, retrieve_all_reuters


def _get_topics(reut) -> List[str]:
    tops = reut.topics.find_all("d")
    pre_strtops = [s.get_text() for s in tops]
    lower_classes = ["money-fx", "dlr", "yen", "dmk"]
    strtops = []
    for t in pre_strtops:
        if t in lower_classes:
            if 'money' not in strtops:
                strtops.append("money")
        else:
            strtops.append(t)
    return strtops


def _get_topics_from_id(newid: str) -> List[str]:
    newid = str(newid)
    for reuter in retrieve_all_relevant_reuters():
        if _get_id(reuter) == newid:
            return _get_topics(reuter)
    print("A document with that id might not be in the training set")


def get_topics_from_list_ids(ids: Dict[str, float]) -> List[str]:
    topics = {}
    with open('document_topic_dictionary.json', 'r') as f:
        topic_document_dictionary = json.load(f)
    for id in ids:
        for t in topic_document_dictionary[id]:
            if t not in topics:
                topics[t] = ids[id]
    return {k: v for k,v in sorted(topics.items(), key=lambda item: item[1], reverse=True)}


def _get_id(reut) -> str:
    return str(reut['newid'])


def _get_content(reut) -> str:
    return reut.content.get_text()

def get_content_from_newid(newid: str) -> str:
    newid = str(newid)
    for reuter in retrieve_all_reuters():
        if _get_id(reuter) == newid:
            return _get_content(reuter)


def _preprocess_content(reut) -> List[str]:
    return preprocessing(_get_content(reut))
