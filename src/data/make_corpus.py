from collections import Counter
import datetime
import nltk
import numpy as np
from pathlib import Path
import pickle
from scipy import spatial
from typing import Dict, List
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

import sys

from src import utils
from src.utils import Segment

### Parameters ###
TIME_DELTA = 60
USE_COS = True


ROOT_DIR = utils.get_project_root()
DATA_DIR = Path.joinpath(ROOT_DIR, 'data')
INTERMEDATE_DATA_DIR = Path.joinpath(DATA_DIR, 'intermediate')




class Corpus:

    def __init__(self, transcript_segments: List[Segment], time_interval: datetime.timedelta) -> List[Segment]:

        self.time_interval: datetime.timedelta = time_interval
        self.documents: List[Segment] = self._create_documents(transcript_segments, time_interval)
        self.num_docs: int = len(self.documents)

        self.vocabulary: List = []
        self.size_vocabulary: int = 0
        self.term2idx: Dict[str:int] = dict()




    def create_vocabulary(self, remove_stop_words: bool) -> None:
        '''
        Tokenizes data and initializes corpus' vocabulary 
        Adds list of tokens to Segment

        Args:
            - remove_stop_words: bool - if true, removes stops words
                defined in NLTK
        
        returns: None
            initializes vocabulary, size_vocabulary, term2idx
        '''
        vocab = set()
        for i, document in enumerate(self.documents):
            tokens = self._tokenize(document.text, remove_stop_words)
            document.tokens = tokens
            self.documents[i] = document
            vocab.update(tokens)

        self.vocabulary = sorted(list(vocab))
        self.size_vocabulary = len(vocab)
        self.term2idx = {word:idx for word, idx in zip(vocab, range(len(vocab)))}


    def create_term_doc_freq_matrix(self) -> None:
        '''
        creates term-document frequency matrix

        Args: None
        
        returns: None
            initializes term_doc_freq_matrix
        '''
        term_freq_docs = []
        for doc in self.documents:
            term_freq_docs.append(Counter(doc.tokens))

        term_document_matrix = np.zeros((self.size_vocabulary, self.num_docs))


        for idx_doc, term_freq in enumerate(term_freq_docs):
            for word, freq in term_freq.items():
                term_document_matrix[self.term2idx[word], idx_doc] = freq

        term_document_matrix /= term_document_matrix.sum(axis=0)

        self.term_doc_freq_matrix = term_document_matrix


    # TODO: adjust constrains so segments fall within time interval instead of just outside of it
    def _create_documents(self, transcript_segments: List[Segment], time_interval: datetime.timedelta) -> List[Segment]:
        '''
        Merges the transcript segments that falls within the time interval

        Args:
            - transcript_segments: List[Segment] - list of trascript segments
            - time_interval: datetime.timedelta - size of time interval
        
        returns: 
            - List[Segment] - new list transcript segments that fall within time interval

        '''
        def merge_documents(docs: List[Segment], id: int) -> Segment:
            return Segment(id=id, beg=docs[0].beg, end=docs[0].end, text=' '.join([seg.text for seg in docs]))

        documents = []
        doc = []
        interval_so_far = datetime.timedelta(seconds=0)
        id = 0
        for i, segment in enumerate(transcript_segments):
            doc.append(segment)
            diff = segment.end - segment.beg
            interval_so_far += diff
            if interval_so_far > time_interval or i == len(transcript_segments) - 1:
                documents.append(merge_documents(doc, id))
                id += 1
                doc = list()
                interval_so_far = datetime.timedelta(seconds=0)


        return documents


    def _tokenize(self, text: str, remove_stop_words: bool=True) -> List[str]:
        '''
        splits text into tokens

        Args:
            - remove_stop_words: bool - if true, removes stops words
                defined in NLTK
        
        returns: List[str]
            list of word tokens
        '''
        tokens = nltk.word_tokenize(text.lower())

        # remove stop words 
        if remove_stop_words:
            tokens = [word for word in tokens if word not in stop_words]

        return tokens


    def calc_similarity_ts(self) -> None:
        '''
        calculates the similarity between sequential document pairs
        uses cosine similarity and jesnen shannoun divergence

        Args: Noe
        
        returns: None
            initializes ts_cos_similarity, ts_divergence_similarity
        '''
        # centered moving text collection
        # texts = []
        # num_segs = 3
        # for i in range(1, len(documents)-1):
        #     tokens = nltk.word_tokenize(' '.join([documents[j].text.lower() for j in range(i-1, i+2)]))
        #     texts.append(tokens)

        # print(texts)


        f_cos = lambda v1, v2: 1 - spatial.distance.cosine(v1, v2)
        f_div = lambda v1, v2: spatial.distance.jensenshannon(v1, v2)

        self.ts_cos_similarity = np.zeros(self.num_docs - 1)
        self.ts_divergence_similarity = np.zeros(self.num_docs - 1)

        similarities = []
        for i in range(1, self.num_docs):
            v1, v2 = self.term_doc_freq_matrix[:, i-1], self.term_doc_freq_matrix[:, i]
            self.ts_cos_similarity[i-1] = f_cos(v1, v2)
            self.ts_divergence_similarity[i-1] = f_div(v1, v2)
        




def main():
    with open(Path.joinpath(INTERMEDATE_DATA_DIR, 'transcripts.pkl'), 'rb') as f:
        transcripts = pickle.load(f)
    
    interval = datetime.timedelta(seconds=TIME_DELTA)



    for transcript_name, transcript_segments in transcripts.items():
    # transcript_segments = transcripts['04_week-4/02_week-4-lessons/01_lesson-4-1-probabilistic-retrieval-model-basic-idea']


        corpus = Corpus(transcript_segments, interval)
        corpus.create_vocabulary(remove_stop_words=True)
        corpus.create_term_doc_freq_matrix()
        corpus.calc_similarity_ts()

        transcripts[transcript_name] = corpus


    # output data as pickle file
    print('saving corpuses')
    with open(Path.joinpath(INTERMEDATE_DATA_DIR, 'corpus.pkl'), 'wb') as f:
        pickle.dump(transcripts, f)

if __name__ == '__main__':
    main()
