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
from sklearn.decomposition import TruncatedSVD
from scipy.stats import randint as sp_randint
import re
# Models to try
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

import utils

# Load the punkt tokenizer used for splitting documents into sentences
tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')

data_set = utils.load_dirs_custom([
    './SENSITIVE_DATA/html-tagged',
    './PERSONAL_DATA/html-tagged',
    './NON_PERSONAL_DATA'
])

doc_train, doc_test, = utils.document_test_train_split(
    data_set, 0.20
)

X_train, y_train = utils.convert_docs_to_lines(doc_train)
X_test, y_test = utils.convert_docs_to_lines(doc_test)

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
min_word_count = 40   # Minimum word count                        
num_workers = 3       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words
if not os.path.exists(model_name): 
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
else:
    model = Word2Vec.load(model_name)
    
del train_sentences


# After here I get absolutely messed up and have no idea what is going on and why things do not work. I get an invalid warning on true divides, and if you try to run the random forest it just gets killed.
def make_feature_vec(words, model, num_features):
    """
    Average the word vectors for a set of words
    """
    feature_vec = np.zeros((num_features,),dtype="float32")  # pre-initialize (for speed)
    nwords = 0
    index2word_set = set(model.wv.index2word)  # words known to the model

    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            feature_vec = np.add(feature_vec,model[word])
    if(nwords != 0):
        feature_vec = np.divide(feature_vec, nwords)
    else:
        feature_vec = np.divide(feature_vec, 1)
    return feature_vec


def get_avg_feature_vecs(documents, model, num_features):
    """
    Calculate average feature vectors for all documents
    """
    counter = 0
    document_feature_vecs = np.zeros((len(documents),num_features), dtype='float32')  # pre-initialize (for speed)
    
    for document in documents:
        document_feature_vecs[counter] = make_feature_vec(document, model, num_features)
        counter = counter + 1
    return document_feature_vecs
 

# calculate average feature vectors for training and test sets
clean_train_documents = []
for document in X_train:
    clean_train_documents.append(document_to_word_list(document, remove_stopwords=True))
trainDataVecs = get_avg_feature_vecs(clean_train_documents, model, num_features)

clean_test_documents = []
for document in X_test:
    clean_test_documents.append(document_to_word_list(document, remove_stopwords=True))
testDataVecs = get_avg_feature_vecs(clean_test_documents, model, num_features)


# ======================================================================================

# Fit a random forest to the training data, using 100 trees
# forest = RandomForestClassifier(n_estimators = 10)

# This was to help with the NaN values apparently
trainDataVecs = Imputer().fit_transform(trainDataVecs)

#print("Fitting a random forest to labeled training data...")
#forest = forest.fit(trainDataVecs, X_train)

#print("Predicting labels for test data..")
#result = forest.predict(testDataVecs)

#print(classification_report(X_test, result))

clf = SGDClassifier(shuffle=True, tol=None, max_iter=1000, loss='hinge', penalty='l2', class_weight='balanced')

params = {
 "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron", "squared_loss", "huber"],
 "penalty": ['l1', 'l2', 'none'],
 "class_weight": [{0:1, 1:1.5, 2:1.75}, {0:1, 1:2, 0:3}, {0:1, 1:3, 0:5},"balanced", None]}
					


print("Training Model")
clf.fit(trainDataVecs, X_train)
print("SGD")

documents_predicted = []
documents_target = []
all_predicted_lines = []
all_target_lines = []
for doc in doc_test:
    predicted_lines = clf.predict(testDataVecs)
    all_predicted_lines += list(predicted_lines)
    all_target_lines += list(doc.targets)

    predicted_doc = utils.classify_doc(predicted_lines)
    documents_predicted.append(predicted_doc)
    documents_target.append(doc.category)


print("Line by Line ")
print("Confusion Matrix: \n{}".format(
    confusion_matrix(all_target_lines, all_predicted_lines)
))

accuracy = fbeta_score(
    all_target_lines,
    all_predicted_lines,
    average=None,
    beta=2
)
print("Accuracy: {}".format(accuracy))


doc_accuracy = fbeta_score(
    documents_target,
    documents_predicted,
    average=None,
    beta=2
)

print("Document Accuracy: {}".format(doc_accuracy))

print("Document Confusion Matrix: \n{}".format(
    confusion_matrix(documents_target, documents_predicted)
))
