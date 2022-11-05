from dataclasses import dataclass
import datetime
import pandas as pd
from pathlib import Path
import pickle
from typing import List



from src import utils
from src.utils import Segment


ROOT_DIR = utils.get_project_root()
DATA_DIR = Path.joinpath(ROOT_DIR, 'data')
DATA_RAW_DIR = Path.joinpath(DATA_DIR, 'raw/cs-410')
INTERMEDATE_DATA_DIR = Path.joinpath(DATA_DIR, 'intermediate')




def format_segment(seg: List[str]) -> Segment:
    ''' 
    Helper function to format a segment from a transcript.

    Converts the segment ID to an int and subtracts one so zero-indexed
    Convervts beginning and end times of segments to datetime's

    Args:
        - seg: text segment of transcript
            - List[0]: segment ID
            - List[1]: beginning time of text segment
            - List[2]: end time of text segment
            - List[3]: text of segment

    Returns:
        Segment dataclass
    '''
    beg, end = seg[1].split(' --> ')
    beg = datetime.datetime.strptime(beg, '%H:%M:%S,%f')
    end = datetime.datetime.strptime(end, '%H:%M:%S,%f')
    text = ' '.join(seg[2:])
       
    return Segment(int(seg[0])-1, beg, end, text)


def process_transcript(file_path: Path) -> List[Segment]:
    '''
    Processes raw transcript file and outputs a list of all
    the text segments in the transcript

    Args:
        - file_path: complete file path and file name of transcript to process

    Returns:
        - list of all text segments in the transcript
    '''
    segments = []
    seg = []
    with open(file_path, 'r') as f:
        for line in f:
            if line != '\n':
                seg.append(line.strip())
                # print(i, 'empty')
            else:
                segments.append(format_segment(seg))
                seg = list()
    return segments






def main():
    file_name = '04_week-4/02_week-4-lessons/01_lesson-4-1-probabilistic-retrieval-model-basic-idea.en.srt'
    file_path = Path.joinpath(DATA_RAW_DIR, file_name)

    transcript_segments = process_transcript(file_path)

    for s in transcript_segments:
        print(s)

    # output file to intermediate folder taking file extension as input
    output_file = lambda ext: Path.joinpath(INTERMEDATE_DATA_DIR, f'transcripts.{ext}')

    # output data as pickle file
    with open(output_file('pkl'), 'wb') as f:
        pickle.dump(transcript_segments, f)

    df = pd.DataFrame(transcript_segments).set_index('id')
    df['file_name'] = file_name
    print(df)
    df.to_csv(output_file('csv'))



if __name__ == '__main__':
    main()