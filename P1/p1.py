#!/usr/bin/env python3.6

from collections import Counter, defaultdict
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
TFIDFS = Counter()
TOKENIZER = RegexpTokenizer(r'[a-zA-Z]+').tokenize
# Can't DFTS.keys() be used instead of TOKENCORPUS??
TOKENCORPUS = set()

def setup():
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
    print('duration:', time()-t)
    print('total unique tokens from corpus', len(DFTS))
    ord_tokens = [(key, val) for key, val in DFTS.items()]
    ord_tokens.sort(key=lambda x: x[1], reverse=True)
    d = Counter()
    for key, value in DFTS.items():
        IDFS[key] = N / value
    for key, value in IDFS.items():
        d[value] += 1
    for key, value in d.items():
        print(f'{value} words with inverse frequency {key}')
    query('republican thank republican')
    # print(DOCS['1960-09-26.txt']['tokens'][:10])
    # print(len(DOCS['1960-09-26.txt']['tokens']))
    # print(len(DOCS['1960-09-26.txt']['frequencies']))
    # c = 0
    # for key,val in DOCS['1960-09-26.txt']['frequencies'].items():
    #     c += 1
    #     print(key, val)
    #     if c > 10: break

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

def make_tfidf(doc):
    pass

def make_dfts(docs):
     # parallelize this
    for doc in FILENAMES:
        # make_dft()
        words = set()
        for word in DOCS[doc]:
            words.add(word)
            # TOKENCORPUS.add(word)
        return words
    for word in TOKENCORPUS:
        dft = 0
        for doc in FILENAMES:
            if word in DOCS[doc]['tokens'].keys():
                dft += 1
        DFTS[word] = dft

def make_dft(doc):
    words = set()
    for word in DOCS[doc]:
        words.add(word)
        # TOKENCORPUS.add(word)
    return words

def del_then_stem_words(words):
    return [STEMMER(word) for word in words if word not in STOPWORDS]

def getidf(token):
    if True:
        return 0
    else:
        return -1
    pass

def getweight(filename, token):
    # goal:
    # return DOCS[filename]['tokens'][token]['wtd']
    return 0
    for doc in DOCS:
        pass
        # compute w-t,d
        wtd = (1 + log10(tftd)) * (log10(N / dft))
    pass

def query(qstring):
    tokens = del_then_stem_words(qstring.lower().split())
    # goal:
    # matches = priorityqueue()
    # for token in tokens:
    matches = {}
    counts = Counter()
    vector = {}
    for token in tokens:
        matches[token] = DFTS[token]
        counts[token] += 1
    for term, freq in counts.items():
        vector[term] = 1 + log10(freq)
    vector = normalize_vector(vector)
    print(vector)
    if True:
        return "test", 0
    else:
        return ('None', 0)

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
    exit(0)
    tests = [
        ("%.12f" % getidf("health"), "0.079181246048"),
        ("%.12f" % getidf("agenda"), "0.363177902413"),
        ("%.12f" % getidf("vector"), "-1.000000000000"),
        ("%.12f" % getidf("reason"), "0.000000000000"),
        ("%.12f" % getidf("hispan"), "0.632023214705"),
        ("%.12f" % getidf("hispanic"), "-1.000000000000"),
        ("%.12f" % getweight("2012-10-03.txt","health"), "0.008528366190"),
        ("%.12f" % getweight("1960-10-21.txt","reason"), "0.000000000000"),
        ("%.12f" % getweight("1976-10-22.txt","agenda"), "0.012683891289"),
        ("%.12f" % getweight("2012-10-16.txt","hispan"), "0.023489163449"),
        ("%.12f" % getweight("2012-10-16.txt","hispanic"), "0.000000000000"),
        ("(%s, %.12f)" % query("health insurance wall street"), \
            "(2012-10-03.txt, 0.033877975254)"),
        ("(%s, %.12f)" % query("particular constitutional amendment"), \
            "(fetch more, 0.000000000000)"),
        ("(%s, %.12f)" % query("terror attack"), \
            "(2004-09-30.txt, 0.026893338131)"),
        ("(%s, %.12f)" % query("vector entropy"), "(None, 0.000000000000)")
    ]

    cases = len(tests)
    fails = 0
    for i, (result, score)in enumerate(tests):
        if result != score:
            fails += 1
            cases -= 1
            print(f'Test {i} failed:')
            print(result, '==', score)
            print()
    print(f'{cases} test cases passed')
    print(f'{fails} test cases failed')

if __name__ == '__main__':
    run_tests()

