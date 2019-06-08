# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:27:59 2019

@author: Nauman Ahmed
"""

import sys
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, naive_bayes, metrics
from hypopt import GridSearch

if __name__ == "__main__":
    list_names = ["train_pos_path", "train_neg_path", "valid_pos_path", \
                  "valid_neg_path", "test_pos_path", "test_neg_path"]
    
    label_names = ["train_labels", "validation_labels", "test_labels"]
    
    set_names = ['training_set', 'validation_set', 'test_set']
    
    file_lists = {key:[] for key in list_names}
    label_lists = {key:[] for key in label_names}
    dataset_lists = {key:[] for key in set_names}
    
    j = 0
    # Open each file and put the entries into a list
    for i in range(1, 7):
        input_path = sys.argv[i]
        
        fd = open(input_path)
        
        for line in fd:
            file_lists[list_names[i - 1]].append(line)
            
            if (i % 2) == 0:
                label_lists[label_names[j]].append(0) # For negative review labels
            else:
                label_lists[label_names[j]].append(1) # For positive review labels
                
        if (i % 2) == 0:
            j += 1
   
        fd.close()
    
    # Merge pos and neg reviews to create consolidated datasets
    dataset_lists[set_names[0]] = file_lists[list_names[0]] + file_lists[list_names[1]]
    dataset_lists[set_names[1]] = file_lists[list_names[2]] + file_lists[list_names[3]]
    dataset_lists[set_names[2]] = file_lists[list_names[4]] + file_lists[list_names[5]]
    
    # Shuffle datasets & labels in unison 
    dataset_lists[set_names[0]], label_lists[label_names[0]] = shuffle(dataset_lists[set_names[0]], label_lists[label_names[0]])
    dataset_lists[set_names[1]], label_lists[label_names[1]] = shuffle(dataset_lists[set_names[1]], label_lists[label_names[1]])
    dataset_lists[set_names[2]], label_lists[label_names[2]] = shuffle(dataset_lists[set_names[2]], label_lists[label_names[2]])
    
    
    # Build vocaculary for transformation to matrix with unigrams
    count_vect = CountVectorizer(analyzer='word', ngram_range = (1, 1))
    count_vect.fit(dataset_lists[set_names[0]] + dataset_lists[set_names[1]])
    
    # Transform datasets to sparse matrix representations
    x_train = count_vect.transform(dataset_lists[set_names[0]])
    x_valid = count_vect.transform(dataset_lists[set_names[1]])
    x_test = count_vect.transform(dataset_lists[set_names[2]])
    
    # Tune naive bayes alpha value using Gridsearch and use the best value for final score
    opt = GridSearch(model = naive_bayes.MultinomialNB(), param_grid = [{'alpha': [1, 2, 3]}])
    opt.fit(x_train, label_lists[label_names[0]], x_valid, label_lists[label_names[1]])
    print('Test Score for Optimized Parameters with unigrams: {:.4f}'.format(opt.score(x_test, label_lists[label_names[2]])))
    
    
    # Build vocaculary for transformation to matrix with bigrams
    count_vect = CountVectorizer(analyzer='word', ngram_range = (2, 2))
    count_vect.fit(dataset_lists[set_names[0]] + dataset_lists[set_names[1]])
    
    # Transform datasets to sparse matrix representations
    x_train = count_vect.transform(dataset_lists[set_names[0]])
    x_valid = count_vect.transform(dataset_lists[set_names[1]])
    x_test = count_vect.transform(dataset_lists[set_names[2]])
    
    # Tune naive bayes alpha value using Gridsearch and use the best value for final score
    opt = GridSearch(model = naive_bayes.MultinomialNB(), param_grid = [{'alpha': [1, 2, 3]}])
    opt.fit(x_train, label_lists[label_names[0]], x_valid, label_lists[label_names[1]])
    print('Test Score for Optimized Parameters with bigrams: {:.4f}'.format(opt.score(x_test, label_lists[label_names[2]])))
    
    
    # Build vocaculary for transformation to matrix with unigrams + bigrams
    count_vect = CountVectorizer(analyzer='word', ngram_range = (1, 2))
    count_vect.fit(dataset_lists[set_names[0]] + dataset_lists[set_names[1]])
    
    # Transform datasets to sparse matrix representations
    x_train = count_vect.transform(dataset_lists[set_names[0]])
    x_valid = count_vect.transform(dataset_lists[set_names[1]])
    x_test = count_vect.transform(dataset_lists[set_names[2]])
    
    # Tune naive bayes alpha value using Gridsearch and use the best value for final score
    opt = GridSearch(model = naive_bayes.MultinomialNB(), param_grid = [{'alpha': [1, 2, 3]}])
    opt.fit(x_train, label_lists[label_names[0]], x_valid, label_lists[label_names[1]])
    print('Test Score for Optimized Parameters with unigrams + bigrams: {:.4f}'.format(opt.score(x_test, label_lists[label_names[2]])))
    
    
    

    
    
    
    
    
    
    