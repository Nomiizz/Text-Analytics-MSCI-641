# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:17:01 2019

@author: Nauman Ahmed
"""

import os
import sys
from sklearn.utils import shuffle
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Activation, Flatten
from keras.regularizers import l2
import ast
from keras.utils import to_categorical


# Load the word embeddings from the embeddings text file

embeddings_dict = {};

fd = open(os.path.join('', 'word2vecEmbeddings.txt'), encoding = "utf-8")
for line in fd:
    vals = line.split()
    word = vals[0]
    
    coeffs = np.asarray(vals[1:])
    embeddings_dict[word] = coeffs
    
fd.close()

# Load the datasets
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
            file_lists[list_names[i - 1]].append(ast.literal_eval(line))
            
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


    # Convert the text samples to integer samples
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(dataset_lists[set_names[0]] + dataset_lists[set_names[1]] + dataset_lists[set_names[2]])
    
    train_sequences = tokenizer_obj.texts_to_sequences(dataset_lists[set_names[0]])
    val_sequences = tokenizer_obj.texts_to_sequences(dataset_lists[set_names[1]])
    test_sequences = tokenizer_obj.texts_to_sequences(dataset_lists[set_names[2]])
    

    # pad sequences
    max_length_arr = [len(s) for s in (train_sequences + val_sequences + test_sequences)]
    max_length = max(max_length_arr)
    
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')

    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')
    
    val_padded = pad_sequences(val_sequences, maxlen=max_length, padding='post', truncating='post')
    
    # Create an embedding matrix containing only the word's in our vocabulary
    # If the word does not have a pre-trained embedding, then randomly initialize the embedding
    embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(tokenizer_obj.word_index)+1, len(coeffs))) # +1 is because the matrix indices start with 0

    for word, i in tokenizer_obj.word_index.items(): # i=0 is the embedding for the zero padding
        try:
            embeddings_vector = embeddings_dict[word]
        except KeyError:
            embeddings_vector = None
        if embeddings_vector is not None:
            embeddings_matrix[i] = embeddings_vector
        
    del embeddings_dict
    
  # Convert 1-D lists to categorical so that softmax can be applied in the output layer
    y_train = to_categorical(np.asarray(label_lists[label_names[0]]))
    y_val = to_categorical(np.asarray(label_lists[label_names[1]]))
    y_test = to_categorical(np.asarray(label_lists[label_names[2]]))
    
    # Build the Model
    
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer_obj.word_index)+1,
                          output_dim=len(coeffs), input_length=max_length,
                          weights = [embeddings_matrix], trainable=False, name='word_embedding_layer'))
    
    model.add(Dense(64, activation='sigmoid', kernel_regularizer= l2(0.001)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(2, activation = 'softmax', kernel_regularizer= l2(0.001), name='output_layer'))
    
    model.summary()
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    model.fit(train_padded, y_train, batch_size=1024, epochs=10, validation_data=(val_padded, y_val))

    score, acc = model.evaluate(test_padded, y_test,
                            batch_size=1024)
    print("Accuracy on the test set = {0:4.3f}".format(acc))
