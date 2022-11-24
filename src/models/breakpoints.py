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


def calc_breakpoints(corpus: Corpus) -> List[int]:
    algo = rpt.Pelt(model="rbf").fit(corpus.ts_cos_similarity)
    std = np.std(corpus.ts_cos_similarity)
    result = algo.predict(pen=std)

    return result

def merge_documents_breakpoints(documents: List[Segment], breakpoints: int) -> List[Segment]:
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
    for transcript_name, corpus_times in corpuses.items():
        topic_transitions_corpuses[transcript_name] = dict()

        for time_delta, corpus in corpus_times.items():
            breakpoints = calc_breakpoints(corpus)
            print(breakpoints)
    

            subtopic_docs = merge_documents_breakpoints(corpus.documents, breakpoints)
            topic_transitions_corpus = Corpus(corpus.vocab, subtopic_docs)
            topic_transitions_corpus.create_term_doc_freq_matrix()
            topic_transitions_corpus.calc_similarity_ts()

            topic_transitions_corpuses[transcript_name][time_delta] = corpus


            df = pd.DataFrame(subtopic_docs)
            df['file_name'] = transcript_name
            df['time_interval'] = time_delta
            dataframes.append(df)

    output_file = lambda ext: Path.joinpath(INTERMEDATE_DATA_DIR, f'transcripts.{ext}')

    df = pd.concat(dataframes, axis=0)
    df.to_csv(Path.joinpath(INTERMEDATE_DATA_DIR, 'topic_transitions.csv'))
    print(df)
    print(df.groupby(['file_name', 'time_interval']).agg({'end' : ['count', 'max']}))
            
