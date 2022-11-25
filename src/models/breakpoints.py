import datetime
import nltk
from nltk.text import TextCollection
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy import spatial
import seaborn as sns
from matplotlib import pyplot as plt
import ruptures as rpt
from typing import List
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

from src import utils
from src.utils import Segment
from src.data import process_transcripts
from src.data.make_corpus import Corpus, Vocabulary


ROOT_DIR = utils.get_project_root()
DATA_DIR = Path.joinpath(ROOT_DIR, 'data')
DATA_RAW_DIR = Path.joinpath(DATA_DIR, 'raw/cs-410')
INTERMEDATE_DATA_DIR = Path.joinpath(DATA_DIR, 'intermediate')


def calc_breakpoints(time_series: np.array) -> List[int]:
    '''
    calcualates the breakpoints of a time series using Pelt

    Args:
        - time_series: np.array - time series of correlations
    
    returns: List[int] - indexes of time series breakpoitns
    '''
    algo = rpt.Pelt(model="rbf").fit(time_series)
    std = np.std(time_series)
    result = algo.predict(pen=std)

    return result

def merge_documents_breakpoints(documents: List[Segment], breakpoints: int) -> List[Segment]:
    '''
    combine list of documents based on breakpoints

    Args:
        - documents: List[Segment] - list of transcript segments
        - breakpoints: List[int] - indexes of time series breakpoitns
    
    returns: List[Segment] - list of transcript segments
    '''
    subtopic_docs = []
    start_idx = 0
    for id, end_idx in enumerate(breakpoints):
        subtopic_docs.append(utils.merge_documents(documents[start_idx:end_idx], id))
        start_idx = end_idx
    else:
        subtopic_docs.append(utils.merge_documents(documents[end_idx:], id+1))

    return subtopic_docs


if __name__ == '__main__':
    with open(Path.joinpath(INTERMEDATE_DATA_DIR, 'corpuses.pkl'), 'rb') as f:
        corpuses = pickle.load(f)

    # file_name = '04_week-4/02_week-4-lessons/01_lesson-4-1-probabilistic-retrieval-model-basic-idea'
    # corpus_times = corpuses[file_name]
    dataframes = []
    topic_transitions_corpuses = dict()       

    for transcript_name, corpus_times in corpuses.items():    # loop through each transcript
        topic_transitions_corpuses[transcript_name] = dict()

        for time_delta, corpus in corpus_times.items():       # loop through each time interval

            # calc breakpoint and make new document corpus based on them
            breakpoints = calc_breakpoints(corpus.ts_cos_similarity)    
            subtopic_docs = merge_documents_breakpoints(corpus.documents, breakpoints)
            topic_transitions_corpus = Corpus(corpus.vocab, subtopic_docs)
            topic_transitions_corpus.create_term_doc_freq_matrix()
            topic_transitions_corpus.calc_similarity_ts()

            # add to dict
            topic_transitions_corpuses[transcript_name][time_delta] = topic_transitions_corpus

            # creae dataframe of results
            df = pd.DataFrame(subtopic_docs)
            df['file_name'] = transcript_name
            df['time_interval'] = time_delta
            dataframes.append(df)

    # output file to intermediate folder taking file extension as input
    output_file = lambda ext: Path.joinpath(INTERMEDATE_DATA_DIR, f'topic_transitions.{ext}')

    # output data as pickle file
    print('saving corpuses')
    with open(output_file('pkl'), 'wb') as f:
        pickle.dump(topic_transitions_corpuses, f)


    # export dataframe
    df = pd.concat(dataframes, axis=0)
    df.to_csv(output_file('csv'), index=False)
    print(df)
    print(df.groupby(['file_name', 'time_interval']).agg({'end' : ['count', 'max']}))
            