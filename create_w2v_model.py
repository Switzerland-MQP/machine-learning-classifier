'''
TUTORIAL THAT WAS FOLLOWED IS FOUND AT https://www.kaggle.com/jungealexander/word2vec-and-random-forest-classification
'''


from gensim.models import Word2Vec, word2vec
import numpy as np
import os
import nltk
import pandas as pd
#nltk.download('stopwords')
from nltk.corpus import stopwords

import math


# Load the punkt tokenizer used for splitting documents into sentences
tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')

def document_to_word_list(document, remove_stopwords=True):
    
    document_text = re.sub("[^a-zA-Z]"," ", document)
    
    words = document_text.lower().split()
    
    if remove_stopwords:
        stops = set(stopwords.words("german"))
        words = [w for w in words if not w in stops]
    
    return words


def document_to_sentences(document, tokenizer, remove_stopwords=True):
    """
    Split document into list of sentences where each sentence is a list of words.
    Removal of stop words is optional.
    """
    # use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(document.strip())

    # each sentence is furthermore split into words
    sentences = []    
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            sentences.append(document_to_word_list(raw_sentence, remove_stopwords))
            
    return sentences


train_sentences = []  # Initialize an empty list of sentences

for root, dirs, files in os.walk('./TEXTDATA'):
    for file in files:
        full_path = root + '/' + file
        with open(full_path) as f:
            train_sentences += document_to_sentences(f.read(), tokenizer)

model_name = 'train_model'
# Set values for various word2vec parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 5   # Minimum word count                        
num_workers = 3       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words
# Initialize and train the model (this will take some time)
model = word2vec.Word2Vec(train_sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model.save(model_name)
    
del train_sentences

