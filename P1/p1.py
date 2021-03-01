#!/usr/bin/env python3.6

from collections import Counter, defaultdict
from functools import reduce
from math import log10
from multiprocessing import Pool
from queue import PriorityQueue
from time import time

# Adapted from P1_guidelines.ipnb:
# {
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from os import listdir
from os.path import join

# DOCS = defaultdict(lambda: defaultdict(lambda: []))
# DOCS = {'filename': {'tokens': tokenized_doc[], 'frequencies': Counter(tokenized_doc)}}
# STOPWORDS = maketrie(stopwords.words('english'))
# STOPWORDS = set(stopwords.words('english'))
CORPUSDIR = './presidential_debates'
# DFTS is a dict('token': int(doc_freq)} that maps tokens to the qty of docs
# containing that term (ie NOT IDF, just the qty of docs containing the token)
DFTS =  Counter() # PriorityQueue()
DOCS = {}
FILENAMES = ''
IDFS = Counter()
N = float(0)
STEMMER = PorterStemmer().stem
STOP = 'STOP'
STOPWORDS = stopwords.words('english')
# TFIDFS = Counter()
TFIDFS = defaultdict(lambda: Counter())
TOKENIZER = RegexpTokenizer(r'[a-zA-Z]+').tokenize
# Can't DFTS.keys() be used instead of TOKENCORPUS??
# TOKENCORPUS =

def setup():
    global FILENAMES
    global IDFS
    global N
    FILENAMES = listdir(CORPUSDIR)
    N = float(len(FILENAMES))
    t = time()
    with Pool() as p:
        DOCS = dict(p.map(process_document, FILENAMES))
        for tokens in DOCS.values():
            # TOKENCORPUS.update(tokens)
            for token in tokens:
                DFTS[token] += 1
        # # token_sets = set([tokens.keys() for tokens
        # # token_sets = list(p.map(make_token_list_to_set, DOCS.values()))
        # for token_set in token_sets:
        #     TOKENCORPUS.update(token_set)
    # print('parallel duration:', time()-t)
    # print('total unique tokens from corpus', len(DFTS))
    # ord_tokens = [(key, val) for key, val in DFTS.items()]
    # ord_tokens.sort(key=lambda x: x[1], reverse=True)
    for key, value in DFTS.items():
        IDFS[key] = log10(N / value)
    for doc in DOCS:
        TFIDFS[doc] = make_tfidf(DOCS[doc], IDFS)
# }

def process_document(filename):
    with open(join(CORPUSDIR, filename), "r", encoding='UTF-8') as f:
        words = del_then_stem_words(TOKENIZER(f.read().lower()))
        # return (filename, {'tokens': Counter(words)})
        return (filename, Counter(words))
        # return (filename, {'tokens': Counter(doc)}) # , 'tfidf': tfidf})
        # return (filename, {'tokens': doc, 'frequencies': Counter(doc)})
        # DOCS[filename]['tokens'] = doc
        # DOCS[filename]['frequencies'] = Counter(doc)
        # print(len(DOCS.keys()))

def make_token_list_to_set(doc):
    return set(doc.keys())

def make_tfidf(doc, idfs):
    vector = Counter()
    for token, freq in doc.items():
        vector[token] = (1 + log10(freq)) * idfs[token]
    return normalize_vector(vector)

# def make_dfts(docs):
#      # parallelize this
#     for doc in FILENAMES:
#         # make_dft()
#         words = set()
#         for word in DOCS[doc]:
#             words.add(word)
#             # TOKENCORPUS.add(word)
#         return words
#     for word in TOKENCORPUS:
#         dft = 0
#         for doc in FILENAMES:
#             if word in DOCS[doc]['tokens'].keys():
#                 dft += 1
#         DFTS[word] = dft

def make_dft(doc):
    words = set()
    for word in DOCS[doc]:
        words.add(word)
    return words

def make_query_wtd(tokens):

        vector[term] = 1 + log10(freq)

def del_then_stem_words(words):
    return [STEMMER(word) for word in words if word not in STOPWORDS]

def getidf(token):
    global IDFS
    return IDFS[token] if token in IDFS else -1

def getweight(filename, token):
    global TFIDFS
    return TFIDFS[filename][token]
    # goal:
    # return DOCS[filename]['tokens'][token]['wtd']
    # return 0
    # for doc in DOCS:
    #     pass
    #     # compute w-t,d
    #     wtd = (1 + log10(tftd)) * (log10(N / dft))
    # pass

def query(qstring):
    global DFTS # The keys of this dict serve as TOKENCORPUS
    global TFIDFS
    tokens = del_then_stem_words(qstring.lower().split())
    p_list = defaultdict(lambda: PriorityQueue())

    for token in set(tokens):
        if token in DFTS:
            for filename, tfidfvec in TFIDFS.items():
                if token in tfidfvec:
                    p_list[token].put((-1 * tfidfvec[token], filename))

    top10 = defaultdict(lambda: set())

    for token in set(tokens):
        for _ in range(10):
            if not p_list[token].empty():
                top10[token].add(p_list[token].get()[1])
            else:
                break

    # the docs which contain at least one instance of all query tokens
    all_docs = set().union(*(top10.values()))

    need_more_than_10 = False
    if len(all_docs):
        scores = Counter()
        query_vector = normalize_vector(Counter(tokens))

        for filename in all_docs:
            for token, freq in Counter(tokens).items():
                scores[filename] += query_vector[token] * TFIDFS[filename][token]
    else:
        need_more_than_10 = True

    if need_more_than_10:
        return ('fetch more', 0)
    elif len(scores):
        # result = max(scores, key=scores.get)
        result = 'None'
        return (result, scores[result])
    else:
        return ('None', 0)

    # goal:
    # matches = priorityqueue()
    # for token in tokens:
def make_query_vector(tokens):
    matches = Counter()
    counts = Counter()
    vector = Counter()
    for token in tokens:
        matches[token] = DFTS[token]
        counts[token] += 1
    for term, freq in counts.items():
        vector[term] = 1 + log10(freq)
    return normalize_vector(vector)

def normalize_vector(v):
    den = 0
    for value in v.values():
        den += (value ** 2)
    den **= (1/2)
    # v = {(key, (value / den)) for key, value in v.items()}
    for key, value in v.items():
        v[key] = value / den
    return v

def run_tests():
    setup()
    # exit(0)
    tests = [
        (0, "%.12f" % getidf("health"), "0.079181246048"),
        (0, "%.12f" % getidf("agenda"), "0.363177902413"),
        (0, "%.12f" % getidf("vector"), "-1.000000000000"),
        (0, "%.12f" % getidf("reason"), "0.000000000000"),
        (0, "%.12f" % getidf("hispan"), "0.632023214705"),
        (0, "%.12f" % getidf("hispanic"), "-1.000000000000"),
        (0, "%.12f" % getweight("2012-10-03.txt","health"), "0.008528366190"),
        (0, "%.12f" % getweight("1960-10-21.txt","reason"), "0.000000000000"),
        (0, "%.12f" % getweight("1976-10-22.txt","agenda"), "0.012683891289"),
        (0, "%.12f" % getweight("2012-10-16.txt","hispan"), "0.023489163449"),
        (0, "%.12f" % getweight("2012-10-16.txt","hispanic"), "0.000000000000"),
        (0, "(%s, %.12f)" % query("health insurance wall street"), \
            "(2012-10-03.txt, 0.033877975254)"),
        (1, "(%s, %.12f)" % query("particular constitutional amendment"), \
            "(fetch more, 0.000000000000)"),
        (0, "(%s, %.12f)" % query("terror attack"), \
            "(2004-09-30.txt, 0.026893338131)"),
        (0, "(%s, %.12f)" % query("vector entropy"), "(None, 0.000000000000)")
    ]

    cases = 0
    passes = 0
    fails = 0
    for i, (active, result, score)in enumerate(tests):
        if active:
            cases += 1
            if result != score:
                fails += 1
                print(f'Test {i} failed:')
                print(result, '==', score)
                print()
            else:
                passes += 1
    print(f'{cases} test cases run')
    print(f'{passes} test cases passed')
    print(f'{fails} test cases failed')

if __name__ == '__main__':
    run_tests()

