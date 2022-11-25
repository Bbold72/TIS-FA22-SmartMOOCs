from dataclasses import dataclass
import datetime
import itertools
from pathlib import Path
from typing import List


@dataclass
class Segment:
    id: int
    beg: datetime.datetime
    end: datetime.datetime
    text: str
    tokens: List[str] = None


def get_project_root() -> Path:
    return Path(__file__).parents[1]


def merge_documents(docs: List[Segment], id: int) -> Segment:
    '''
    combines the Segments in docs into one segment

    Args:
        - docs: List[Segment] - list of transcript segments to be combined
        - id: int - Unique integer ID of new segment
    
    returns: Segment
    '''
    return Segment(id=id, 
                    beg=docs[0].beg, 
                    end=docs[-1].end, 
                    text=' '.join([seg.text for seg in docs]), 
                    tokens=list(itertools.chain.from_iterable([seg.tokens for seg in docs if seg.tokens is not None]))
                    )


# TODO: adjust constraints so segments fall within time interval instead of just outside of it
def merge_documents_time_interval(documents: List[Segment], time_interval: int) -> List[Segment]:
    '''
    Merges the transcript segments that falls within the time interval

    Args:
        - transcript_segments: List[Segment] - list of trascript segments
        - time_interval: datetime.timedelta - size of time interval
    
    returns: 
        - List[Segment] - new list transcript segments that fall within time interval

    '''
    new_doc_list = []
    doc = []
    interval_so_far = datetime.timedelta(seconds=0)
    time_interval  = datetime.timedelta(seconds=time_interval)
    id = 0
    for i, segment in enumerate(documents):
        doc.append(segment)
        diff = segment.end - segment.beg
        interval_so_far += diff
        if interval_so_far > time_interval or i == len(documents) - 1:
            new_doc_list.append(merge_documents(doc, id))
            id += 1
            doc = list()
            interval_so_far = datetime.timedelta(seconds=0)
    return new_doc_list


def make_segment_label_mapping(documents0: List[Segment], documents1: List[Segment]) -> List[int]:
    # make semgent0 the longer list of Segments
    if len(documents1) > len(documents0):
        documents0, documents1 = documents1, documents0

    label = 0 
    label_map = []
    for segment in documents0:
        if segment.end <= documents1[label].end:
            label_map.append(label)
        else:
            label += 1
            label_map.append(label)
    
    return label_map

common_n_grams = (
                    'probabilistic latent semantic analysis',
                    'vector space model',
                    'natural language processing',
                    'maximum likelihood estimate',
                    'background language model',
                    'maximum likelihood estimator',
                    'contextual text mining',
                    'unigram language model',
                    'inverse document frequency',
                    'naive bayes classifier',
                    'language model',
                    'text mining',
                    'vector space',
                    'text retrieval',
                    'search engines',
                    'machine learning',
                    'time series',
                    'mutual information',
                    'conditional entropy',
                    'information retrieval',
                    'web search',
                    'background model',
                    'opinion mining',
                    'naive bayes'
                )