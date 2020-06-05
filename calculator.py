from typing import List
import numpy
from scipy.spatial.distance import cosine


def dot(x: List[float], y: List[float]) -> float:
    return numpy.array(x) @ numpy.array(y)


def cdot(x: List[float], y: List[float]) -> List[float]:
    return (numpy.array(x) * numpy.array(y)).tolist()


def cosine_similarity(x: List[float], y: List[float]) -> float:
    return 1 - cosine(x, y)


def centroid(vectors: List[List[float]]) -> List[float]:
    print(len(vectors))
    print(len(vectors[0]))
    a = numpy.array(vectors)
    m = numpy.average(a, axis=0)
    return m.tolist()