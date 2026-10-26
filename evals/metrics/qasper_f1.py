import re
import string
from collections import Counter

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    """Normalize text the same way as SQuAD's answer F1."""
    s = s.lower()                         
    s = s.translate(_PUNCT_TABLE)          
    s = _ARTICLES.sub(" ", s)             
    return " ".join(s.split())   


def score_answer(prediction:str,gold:str): 

    pred_norm = normalize_answer(prediction)
    pred_token = pred_norm.split()
    gold_norm = normalize_answer(gold)
    gold_token = gold_norm.split()


    if (len(pred_token) == 0) and (len(gold_token) == 0):
        return 1

    

    overlap_token = Counter(pred_token) & Counter(gold_token)
    overlap = sum(overlap_token.values())
    
    if overlap == 0:
        return 0

    precision =  overlap/len(pred_token)

    recall =  overlap/len(gold_token)

    f1_score = (2*precision*recall)/(precision+recall)

    return f1_score


def compare_scores(prediction:str,gold_set:list): 

    max_f1_score = 0 

    for gold in gold_set: 

        f1_score = score_answer(prediction,gold)
        max_f1_score = max(max_f1_score,f1_score)

    return max_f1_score


if __name__ == "__main__": 

    print(normalize_answer("state-of-the-art"))