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
from sklearn.metrics import silhouette_score
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

    # only return breakpoints that are between endpoints
    return [b for b in result if b != 0 and b != len(time_series)]
    

def merge_documents_breakpoints(documents: List[Segment], breakpoints: int) -> List[Segment]:
    '''
    combine list of documents based on breakpoints

    Args:
        - documents: List[Segment] - list of transcript segments
        - breakpoints: List[int] - indexes of time series breakpoitns
    
    returns: List[Segment] - list of transcript segments
    '''
    # subtopic_docs = []
    # start_idx = 0
    # for id, end_idx in enumerate(breakpoints):
    #     subtopic_docs.append(utils.merge_documents(documents[start_idx:end_idx], id))
    #     start_idx = end_idx
    # else:
    #     subtopic_docs.append(utils.merge_documents(documents[end_idx:], id+1))

    # return subtopic_docs
    # if no breakpoints, merge all segments together
    start_idx = 0
    if len(breakpoints) == 0:
        subtopic_docs = [utils.merge_documents(documents[start_idx:], 0)]

    else:
        subtopic_docs = []
        for id, break_idx in enumerate(breakpoints):
            subtopic_docs.append(utils.merge_documents(documents[start_idx:break_idx+1], id))
            start_idx = break_idx + 1
        else:
            subtopic_docs.append(utils.merge_documents(documents[start_idx:], id+1))

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

        # divide raw transcript into 10 second intervals
        base_corpus = Corpus(corpus_times[60].vocab, utils.merge_documents_time_interval(corpus_times[60].vocab.transcript_segements, 10))
        base_corpus.create_term_doc_freq_matrix()

        # create a mapping from 10 second interval to 60 second intervals
        naive_time_label_map = utils.make_segment_label_mapping(base_corpus.documents, corpus_times[60].documents)
        naive_score = silhouette_score(base_corpus.term_doc_freq_matrix.T, naive_time_label_map) # naive, 60 second topic transition score
        corpus_times['base_60'] = corpus_times[60]
        for time_delta, corpus in corpus_times.items():                            # loop through each time interval

            # calc breakpoint and make new document corpus based on them
            if time_delta == 'base_60':
                subtopic_docs = corpus.documents
            else:
                breakpoints = calc_breakpoints(corpus.ts_cos_similarity)    
                subtopic_docs = merge_documents_breakpoints(corpus.documents, breakpoints)
               
            topic_transitions_corpus = Corpus(corpus.vocab, subtopic_docs)
            topic_transitions_corpus.create_term_doc_freq_matrix()

            # evaluate
            # create a mapping from 10 second intervals to model predicted topic transitions
            if breakpoints:
                topic_transitions_corpus.breakpoints: List[int] = breakpoints
                subtopic_label_map = utils.make_segment_label_mapping(base_corpus.documents, topic_transitions_corpus.documents)
                model_score = silhouette_score(base_corpus.term_doc_freq_matrix.T, subtopic_label_map)   # model topic transition score
            else:
                model_score = None 

            # calculate silhouettes scores
            if transcript_name == '04_week-4/02_week-4-lessons/01_lesson-4-1-probabilistic-retrieval-model-basic-idea':
                print(time_delta, naive_score, model_score, breakpoints)
                print([', '.join(word for word, count in doc_count.most_common(5)) for doc_count in topic_transitions_corpus.doc_term_freq])


            # add to dict
            topic_transitions_corpus.naive_silouhette_score: float = naive_score
            topic_transitions_corpus.model_silouhette_score: float = model_score
            topic_transitions_corpuses[transcript_name][time_delta] = topic_transitions_corpus
            # creae dataframe of results
            df = pd.DataFrame(subtopic_docs)
            df['top_5_tokens'] = [', '.join(word for word, count in doc_count.most_common(5)) for doc_count in topic_transitions_corpus.doc_term_freq]
            df['file_name'] = transcript_name
            df['time_interval'] = time_delta
            df['silhouette_score'] = model_score
            dataframes.append(df)

    

    # output file to intermediate folder taking file extension as input
    output_file = lambda ext: Path.joinpath(INTERMEDATE_DATA_DIR, f'topic_transitions.{ext}')

    # output data as pickle file
    print('saving corpuses')
    with open(output_file('pkl'), 'wb') as f:
        pickle.dump(topic_transitions_corpuses, f)


    # export dataframe
    df = pd.concat(dataframes, axis=0)
    df['best'] = df.groupby(['file_name'])['silhouette_score'].transform(max) == df['silhouette_score']
    # df['best'] = np.where(df['best'] == df['silhouette_score'],
    #                         1,
    #                         0
    #                         )

    df.to_csv(output_file('csv'), index=False)
    print(df)
    print(df.groupby(['file_name', 'time_interval']).agg({'end' : ['count', 'max']}))
            
