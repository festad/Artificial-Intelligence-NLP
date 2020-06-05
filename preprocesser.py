from typing import List
from nltk.tokenize import regexp_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def preprocessing(text: str) -> List[str]:
    return _filter_out_stopwords(
        _lemmatize(
            _lower_words(
                _separate_text_into_words(text))))


def _separate_text_into_words(text: str) -> List[str]:
    regex = r'\w+'
    return regexp_tokenize(text, regex)


def _lemmatize(words: List[str]) -> List[str]:
    return [WordNetLemmatizer().lemmatize(w) for w in words]


def _filter_out_stopwords(words: List[str]) -> List[str]:
    return [w for w in words if w not in stopwords.words('english')]


def _lower(word: str) -> str:
    return word.lower()


def _lower_words(words: List[str]) -> List[str]:
    return [_lower(w) for w in words]
