# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:32:31 2019

@author: Nauman Ahmed
"""

import random
import re
import multiprocessing

from gensim.models import Word2Vec

# Setup logging
import logging 
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

# Load and tokenize the data

fd = open('pos.txt')
data = list()

for line in fd:
    tokenStr = re.split('(\W)', line)
    tokenStr = [x.lower() for x in tokenStr if x not in '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n ']
    data.append(tokenStr)
    
fd.close()

fd = open('neg.txt')

for line in fd:
    tokenStr = re.split('(\W)', line)
    tokenStr = [x.lower() for x in tokenStr if x not in '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n ']
    data.append(tokenStr)
    
fd.close()

random.shuffle(data)

# Setup Gensim word2vec

cores = multiprocessing.cpu_count() # Count the number of cores in the computer

w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)

# Build word2vec vocabulary

w2v_model.build_vocab(data, progress_per=10000)

# Train the word2vec

w2v_model.train(data, total_examples=w2v_model.corpus_count, epochs=10, report_delay=1)

w2v_model.init_sims(replace=True)

# Get similarities

print('Good: {}'.format(w2v_model.wv.most_similar(positive=["good"], topn=20)))
print("")
print('Bad: {}'.format(w2v_model.wv.most_similar(positive=["bad"], topn=20)))



