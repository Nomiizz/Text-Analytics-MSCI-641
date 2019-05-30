# -*- coding: utf-8 -*-
"""
Created on Wed May 29 01:35:54 2019

@author: Nauman Ahmed
"""

import random
import re
from nltk.corpus import stopwords
import sys
import numpy as np

if __name__ == "__main__":
    input_path = sys.argv[1]

fd = open(input_path)

data = list()
data_noStopWords = list()

stop_words = set(stopwords.words('english'))

# Read each line, tockenize and remove unwanted characters. Also remove stop words for seperate data-set
for line in fd:
    tokenStr = re.split('(\W)', line)
    tokenStr = [x.lower() for x in tokenStr if x not in '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n ']
    data.append(tokenStr)
    
    tokenStrNSW = [x for x in tokenStr if x not in stop_words]
    data_noStopWords.append(tokenStrNSW)
    
fd.close()

# Shuffle data and split into train, val and test sets
random.shuffle(data)

train_list = data[:int(len(data)*0.8)]
val_list = data[int(len(data)*0.8):int(len(data)*0.9)]
test_list = data[int(len(data)*0.9):]

random.shuffle(data_noStopWords)

train_list_no_stopword = data_noStopWords[:int(len(data_noStopWords)*0.8)]
val_list_no_stopword = data_noStopWords[int(len(data_noStopWords)*0.8):int(len(data_noStopWords)*0.9)]
test_list_no_stopword = data_noStopWords[int(len(data_noStopWords)*0.9):]

# Save to csv files
np.savetxt("train.csv", train_list, delimiter=",", fmt='%s')
np.savetxt("val.csv", val_list, delimiter=",", fmt='%s')
np.savetxt("test.csv", test_list, delimiter=",", fmt='%s')

np.savetxt("train_no_stopword.csv", train_list_no_stopword,
           delimiter=",", fmt='%s')
np.savetxt("val_no_stopword.csv", val_list_no_stopword,
           delimiter=",", fmt='%s')
np.savetxt("test_no_stopword.csv", test_list_no_stopword,
           delimiter=",", fmt='%s')