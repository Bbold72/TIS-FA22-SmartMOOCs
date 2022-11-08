from collections import namedtuple
from dataclasses import dataclass
import datetime
import nltk
from nltk.text import TextCollection
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy import spatial
import seaborn as sns
from typing import List
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

import sys

from src import utils
from src.utils import Segment

### Parameters ###
TIME_DELTA = 30
USE_COS = True


ROOT_DIR = utils.get_project_root()
DATA_DIR = Path.joinpath(ROOT_DIR, 'data')
INTERMEDATE_DATA_DIR = Path.joinpath(DATA_DIR, 'intermediate')




def merge_documents(docs: List[Segment], id: int) -> Segment:
    return Segment(id=id, beg=docs[0].beg, end=docs[0].end, text=' '.join([seg.text for seg in docs]))


def create_documents(transcript_segments: List[Segment], time_interval: datetime.timedelta) -> List[Segment]:
    documents = []
    doc = []
    interval_so_far = datetime.timedelta(seconds=0)
    id = 0
    for segment in transcript_segments:
        doc.append(segment)
        diff = segment.end - segment.beg
        interval_so_far += diff
        if interval_so_far > time_interval:
            documents.append(merge_documents(doc, id))
            id += 1
            doc = list()
            interval_so_far = datetime.timedelta(seconds=0)
    else:
        if doc:
            documents.append(merge_documents(doc, id))
    print(doc)

    return documents



def calc_similarity_ts(documents: List[Segment], use_cos: bool=True):
   # texts = [nltk.word_tokenize(doc.text.lower()) for doc in documents]
    texts = [list(set(nltk.word_tokenize(doc.text.lower())) - stop_words) for doc in documents]


    # centered moving text collection
    # texts = []
    # num_segs = 3
    # for i in range(1, len(documents)-1):
    #     tokens = nltk.word_tokenize(' '.join([documents[j].text.lower() for j in range(i-1, i+2)]))
    #     texts.append(tokens)

    # print(texts)


    text_collection = TextCollection(texts)
    # for word in texts[1].split():
    #     print(word)
    # print()
    # for word in texts[2].split():
    #     print(word)
    print('\n'*5)
    print([w for w in text_collection.vocab()])
    print()
    print(texts[1], len(texts[1]))
    make_word_vec = lambda doc_id: np.array([text_collection.tf(word, texts[doc_id]) for word in text_collection.vocab()])
    vec1 = make_word_vec(1)
    vec2 = make_word_vec(2)
    print(vec1)
    print(vec2)
    result = 1 - spatial.distance.cosine(vec1, vec2)
    print(result)

    if use_cos:
        f = lambda v1, v2: 1 - spatial.distance.cosine(v1, v2)
    else:
        f = lambda v1, v2: spatial.distance.jensenshannon(v1, v2)

    make_word_vec = lambda doc_id: np.array([text_collection.tf(word, texts[doc_id]) for word in text_collection.vocab()])


    similarities = []
    for i in range(1, len(texts)):
        vec1 = make_word_vec(i-1)
        vec2 = make_word_vec(i)

        similarities.append(f(vec1, vec2))
    
    return similarities




def main():
    with open(Path.joinpath(INTERMEDATE_DATA_DIR, 'transcripts.pkl'), 'rb') as f:
        transcripts = pickle.load(f)
    for key in transcripts.keys():
        print(key)
    transcript_segments = transcripts['04_week-4/02_week-4-lessons/01_lesson-4-1-probabilistic-retrieval-model-basic-idea']

    interval = datetime.timedelta(seconds=TIME_DELTA)
    s = transcript_segments[-1]
    diff = s.end - s.beg
    print(diff)
    print(diff < interval)

    documents = create_documents(transcript_segments, interval)

    for i in range(3):
        print(documents[i])
    print(documents[-1])


    # for i, doc in enumerate(documents):
    #     documents[i] = Segment(id=i, beg=doc[0].beg, end=doc[0].end, text=' '.join([seg.text for seg in doc]))


    similarities = calc_similarity_ts(documents, use_cos=USE_COS)

    print(similarities)

    from matplotlib import pyplot as plt
    plt.plot([i*TIME_DELTA for i in range(len(similarities))], similarities)
    plt.ylim([0, 1])
    plt.show()


if __name__ == '__main__':
    main()
