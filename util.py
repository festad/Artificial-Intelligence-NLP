from typing import Dict


def get_first_n_instances_from_top_score(score: Dict[str, float], n: int) -> Dict[str, float]:
    k = 0
    top_n_score = {}
    for result in score:
        top_n_score[result] = score[result]
        k += 1
        if k >= n:
            return {k: v for k,v in sorted(top_n_score.items(), key=lambda item: item[1], reverse=True)}