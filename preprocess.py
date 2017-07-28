import csv
import jieba
import csv
import jieba.analyse
from bs4 import BeautifulSoup
from collections import defaultdict

with open('offsite-test-material/offsite-tagging-training-set.csv', 'r') as f:
    file_reader = csv.reader(f, delimiter=',', quotechar='"')
    next(file_reader)
    file_columns = [(int(_[0]),_[1], _[2]) for _ in file_reader] #id, class, text

labeldicts_small = {_: defaultdict(float) for _ in {_[1] for _ in file_columns}}
labeldicts_large = {_: defaultdict(float) for _ in {_[1] for _ in file_columns}}
docdict = defaultdict(float)
longdocdict = defaultdict(float)

def max_if_exists(token, classdicts, docdict):
    max_occurence = max([classdict[token] for classdict in classdicts if token in classdict])
    total_occurrence = docdict[token]
    return max_occurence/total_occurrence, max_occurence, total_occurrence


for training in file_columns:
    soup = BeautifulSoup(training[2], 'html5lib')
    text_only = soup.get_text()
    tokens = jieba.cut(text_only, cut_all=True)
    long_tokens = jieba.cut(text_only, cut_all=True)
    for token in tokens:
        labeldicts_small[training[1]][token] += 1
        docdict[token] += 1
    for token in long_tokens:
        labeldicts_large[training[1]][token] += 1
        longdocdict[token] += 1

short_classdicts = labeldicts_small.values()
long_classdicts = labeldicts_large.values()
maxfreq_short = {key: max_if_exists(key, short_classdicts, docdict) for key in docdict.keys()}
maxfreq_long = {key: max_if_exists(key, long_classdicts, longdocdict) for key in longdocdict.keys()}

highest_freq_list = []
for training in file_columns:
    soup = BeautifulSoup(training[2], 'html5lib')
    text_only = soup.get_text()
    tokens = jieba.cut(text_only, cut_all=True)
    long_tokens = jieba.cut(text_only, cut_all=True)
    highest_freq_short = max([(_, maxfreq_short[_][0], maxfreq_short[_][2]) for _ in tokens if maxfreq_short[_][2] >20], key=lambda _: _[1])
    highest_freq_long = max([(_, maxfreq_long[_][0], maxfreq_long[_][2]) for _ in long_tokens if maxfreq_long[_][2] >20], key=lambda _: _[1])
    highest_freq_list.append((training[0], highest_freq_long, highest_freq_short))


print(sorted([_  for _ in maxfreq_long.items() if _[1][1] > 10], key=lambda _: -_[1][0])[:1000])
print(len([_ for _ in maxfreq_long.items() if _[1][1] > 10]),
      len([_ for _ in maxfreq_long.items() if _[1][1] > 100]),
      len([_ for _ in maxfreq_long.items() if _[1][0] <1]),
      len(maxfreq_long))

