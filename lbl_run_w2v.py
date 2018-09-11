'''
Currently does not work. Not sure why as of now
'''


from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from scipy.stats import randint as sp_randint
from sklearn.datasets import load_files
from sklearn.decomposition import PCA

# Models to try
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier

from sklearn import metrics
import utils

documents = utils.load_dirs_custom([
    './SENSITIVE_DATA/html-tagged',
    './PERSONAL_DATA/html-tagged',
    './NON_PERSONAL_DATA'
])

documents = utils.n_gram_documents(documents, 2)

doc_train, doc_test, = utils.document_test_train_split(
    documents, 0.20
)

print("Doc train: ", len(doc_train))
print("Doc test: ", len(doc_test))
X_train, y_train = utils.convert_docs_to_lines(doc_train)
X_test, y_test = utils.convert_docs_to_lines(doc_test)


model = Word2Vec.load('train_model')
w2v = dict(zip(model.wv.index2word, model.wv.syn0))


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(word2vec.items())))
        
    def fit(self, X, y):
        return self
        
    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec.items())))

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


# pre_fit = TfidfEmbeddingVectorizer(w2v)
# pre_fit.fit(X_train, y_train)

# X_train_w2v = pre_fit.transform(X_train)
# X_test_w2v = pre_fit.transform(X_test)

clf = Pipeline([('w2v', TfidfEmbeddingVectorizer(w2v)), ('clf', SGDClassifier(loss='hinge', penalty='none', learning_rate='optimal', alpha=1e-4, epsilon=0.1, max_iter=1000, tol=None, shuffle=True))])


clf.fit(X_train, y_train)

print("W2V with SGD")

documents_predicted = []
documents_target = []
all_predicted_lines = []
all_target_lines = []
for doc in doc_test:
    predicted_lines = clf.predict(doc.data)
    all_predicted_lines += list(predicted_lines)
    all_target_lines += list(doc.targets)

    predicted_doc = utils.classify_doc(predicted_lines)
    documents_predicted.append(predicted_doc)
    documents_target.append(doc.category)


print("Line by Line ")
total_accuracy = np.mean(predicted_lines == doc.category)
print("Total Accuracy:")
print(total_accuracy)

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
total_doc_accuracy = np.mean(predicted_doc == doc.targets)
print("Total Accuracy:")
print(total_doc_accuracy)
print("Document Confusion Matrix: \n{}".format(
    confusion_matrix(documents_target, documents_predicted)
))
