from collections import Counter
import datetime
import nltk
from nltk.stem import PorterStemmer
import numpy as np
from pathlib import Path
import pickle
import re
from scipy import spatial
import string
from typing import Dict, List
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

import sys

from src import utils
from src.utils import Segment



ROOT_DIR = utils.get_project_root()
DATA_DIR = Path.joinpath(ROOT_DIR, 'data')
INTERMEDATE_DATA_DIR = Path.joinpath(DATA_DIR, 'intermediate')

class Vocabulary:

    def __init__(self, transcript_segments: List[Segment], remove_stop_words: bool = True, combine_ngrams: bool = True, stem_words: bool= True) -> None:
        '''
        Create unique set of tokens from raw transcript segments.

        Args:
            - remove_stop_words: bool - if true, removes stops words defined in NLTK
            - combine_ngrams: bool - if true, combine common n-grams defined in utils
            - stem_words: bool - if true, stem words using Porter stemmer
        
        returns: List[str]
            list of word tokens
        '''
        self.transcript_segements = transcript_segments
        self.vocabulary: List = []
        self.size: int = 0
        self.term2idx: Dict[str:int] = dict()

        self._remove_stop_words = remove_stop_words
        self._combine_ngrams = combine_ngrams
        self._stem_words = stem_words

        self._create_vocabulary()


    def _tokenize(self, text: str) -> List[str]:
        '''
        splits text into tokens

        Args:
            - remove_stop_words: bool - if true, removes stops words defined in NLTK
        
        returns: List[str]
            list of word tokens
        '''
        text = text.lower()

        # remove punctuation
        text = re.sub(r"[a-z]-[a-z]", ' ', text)
        text = text.strip().translate(str.maketrans('', '', string.punctuation))
        if 'bits' in text:
            text = re.sub(r"\b\d+[s]{0,1}\b", 'NUMBER', text)   # remove numbers

        if self._combine_ngrams:
            for ngram in utils.common_n_grams:
                text = text.replace(ngram, ngram.replace(' ', '_'))


        tokens = nltk.word_tokenize(text)

        # remove stop words 
        if self._remove_stop_words:
            tokens = [word for word in tokens if word not in stop_words]


        if self._stem_words:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]

        return tokens


    def _create_vocabulary(self) -> None:
        '''
        Tokenizes data and initializes corpus' vocabulary 
        Adds list of tokens to Segment

        Args: None
        
        returns: None
            initializes vocabulary, size, term2idx
        '''
        vocab = set()
        for i, document in enumerate(self.transcript_segements):
            tokens = self._tokenize(document.text)
            document.tokens = tokens
            self.transcript_segements[i] = document
            vocab.update(tokens)

        self.vocabulary = sorted(list(vocab))
        self.size = len(vocab)
        self.term2idx = {word:idx for word, idx in zip(vocab, range(len(vocab)))}



class Corpus:

    def __init__(self, vocab: Vocabulary, documents: List[Segment]) -> None:

        self.documents: List[Segment] = documents
        self.num_docs: int = len(self.documents)
        self.vocab = vocab


    def create_term_doc_freq_matrix(self) -> None:
        '''
        creates term-document frequency matrix

        Args: None
        
        returns: None
            initializes term_doc_freq_matrix
        '''
        self.doc_term_freq: List[Counter] = []
        for doc in self.documents:
            self.doc_term_freq.append(Counter(doc.tokens))
        term_document_matrix = np.zeros((self.vocab.size, self.num_docs))


        for idx_doc, term_freq in enumerate(self.doc_term_freq):
            for word, freq in term_freq.items():
                term_document_matrix[self.vocab.term2idx[word], idx_doc] = freq
        term_document_matrix /= term_document_matrix.sum(axis=0)

        self.term_doc_freq_matrix = term_document_matrix


    def calc_similarity_ts(self) -> None:
        '''
        calculates the similarity between sequential document pairs
        uses cosine similarity and jesnen shannoun divergence

        Args: None
        
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

        for i in range(1, self.num_docs):
            v1, v2 = self.term_doc_freq_matrix[:, i-1], self.term_doc_freq_matrix[:, i]
            self.ts_cos_similarity[i-1] = f_cos(v1, v2)
            self.ts_divergence_similarity[i-1] = f_div(v1, v2)
        


def main():
    TIME_DELTAS = [30, 45, 60]
    with open(Path.joinpath(INTERMEDATE_DATA_DIR, 'transcripts.pkl'), 'rb') as f:
        transcripts = pickle.load(f)
    
    corpuses = dict()
    for transcript_name, transcript_segments in transcripts.items():
        corpuses[transcript_name] = dict()
        vocab = Vocabulary(transcript_segments, remove_stop_words=True, combine_ngrams=True, stem_words=True)  

        for TIME_DELTA in TIME_DELTAS:
        # transcript_segments = transcripts['04_week-4/02_week-4-lessons/01_lesson-4-1-probabilistic-retrieval-model-basic-idea']

            documents = utils.merge_documents_time_interval(vocab.transcript_segements, TIME_DELTA)
            corpus = Corpus(vocab, documents)

            corpus.create_term_doc_freq_matrix()
            corpus.calc_similarity_ts()
            corpuses[transcript_name][TIME_DELTA] = corpus


    # output data as pickle file
    print('saving corpuses')
    with open(Path.joinpath(INTERMEDATE_DATA_DIR, f'corpuses.pkl'), 'wb') as f:
        pickle.dump(corpuses, f)


if __name__ == '__main__':
    main()
