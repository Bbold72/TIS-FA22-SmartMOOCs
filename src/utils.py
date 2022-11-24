from dataclasses import dataclass
import datetime
from pathlib import Path
from typing import List


@dataclass
class Segment:
    id: int
    beg: datetime.datetime
    end: datetime.datetime
    text: str


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
    return Segment(id=id, beg=docs[0].beg, end=docs[-1].end, text=' '.join([seg.text for seg in docs]))
