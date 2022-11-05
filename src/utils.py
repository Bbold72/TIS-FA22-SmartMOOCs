from dataclasses import dataclass
import datetime
from pathlib import Path


@dataclass
class Segment:
    id: int
    beg: datetime.datetime
    end: datetime.datetime
    text: str


def get_project_root() -> Path:
    return Path(__file__).parents[1]
