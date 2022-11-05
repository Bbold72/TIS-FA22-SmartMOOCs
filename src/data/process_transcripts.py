from collections import namedtuple
import datetime
import nltk
from nltk.text import TextCollection
import numpy as np
import os
from pathlib import Path
from scipy import spatial
import seaborn as sns
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

import sys

from src import utils

ROOT_DIR = utils.get_project_root()
DATA_DIR = Path.joinpath(ROOT_DIR, 'data')
DATA_RAW_DIR = Path.joinpath(DATA_DIR, 'raw')

### Parameters ###
TIME_DELTA = 30

segments = []
seg = []
file_name = Path.joinpath(DATA_RAW_DIR,'cs-410/04_week-4/02_week-4-lessons/01_lesson-4-1-probabilistic-retrieval-model-basic-idea.en.srt')

with open(file_name, 'r') as f:
    for line in f:
        if line != '\n':
            seg.append(line.strip())
            # print(i, 'empty')
        else:
            segments.append(seg)
            seg = list()
for s in segments:
    print(s)
Segment = namedtuple('Segment', ['id', 'beg', 'end', 'text'])
for i, seg in enumerate(segments):
    beg, end = seg[1].split(' --> ')
    beg = datetime.datetime.strptime(beg, '%H:%M:%S,%f')
    end = datetime.datetime.strptime(end, '%H:%M:%S,%f')
    text = ' '.join(seg[2:])
    segments[i] = Segment(int(seg[0])-1, beg, end, text)

for s in segments:
    print(s)

interval = datetime.timedelta(seconds=TIME_DELTA)
s = segments[-1]
diff = s.end - s.beg
print(diff)
print(diff < interval)

documents = []
doc = []
interval_so_far = datetime.timedelta(seconds=0)
for seg in segments:
    doc.append(seg)
    diff = seg.end - seg.beg
    interval_so_far += diff
    if interval_so_far > interval:
        documents.append(doc)
        doc = list()
        interval_so_far = datetime.timedelta(seconds=0)
else:
    if doc:
        documents.append(doc)
print(doc)

for i in range(3):
    print(documents[i])
print(documents[-1])

for i, doc in enumerate(documents):
    documents[i] = Segment(id=i, beg=doc[0].beg, end=doc[0].end, text=' '.join([seg.text for seg in doc]))

for d in documents:
    print(d)

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

similarities = []
for i in range(len(texts)):
    vec1 = make_word_vec(i-1)
    vec2 = make_word_vec(i)

    result = 1 - spatial.distance.cosine(vec1, vec2)
    # result = 1 - spatial.distance.jensenshannon(vec1, vec2)
    

    similarities.append(result)
print(similarities)

from matplotlib import pyplot as plt
plt.plot([i*TIME_DELTA for i in range(len(similarities))], similarities)
# plt.ylim([0, 1])
# plt.show()



