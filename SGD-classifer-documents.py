import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint
from sklearn.datasets import load_files
from sklearn.decomposition import PCA

# Models to try
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier

from sklearn import metrics

# Import Data
# Downloaded from http://www.nltk.org/nltk_data/

documents = load_files('TEXTDATA/', shuffle=True)

# Split remainder into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    documents.data, documents.target, test_size=0.40
)


count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

#pca = TruncatedSVD(n_components=20)
#pca.fit(X_train_tfidf)
#X_train_pca = pca.transform(X_train_tfidf)

X_test_count = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_count)
#X_test_pca = pca.transform(X_test_tfidf)

clf = SGDClassifier(tol=None, shuffle=True, loss="log", penalty='l2', alpha=0.001)

parameters_rand = {
#	"loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron", "squared_loss", "huber"],
	#"penalty": ["none", "l1", "l2"],
	#"alpha": [0.0001, 0.0005, 0.001],
	"max_iter": sp_randint(10, 1000)
}

print("Now training model")
n_iter_search = 5
random_search = RandomizedSearchCV(clf, param_distributions=parameters_rand,
	n_iter=n_iter_search,
	n_jobs=-1)

#random_search.fit(X_train_pca, y_train)
random_search.fit(X_train_tfidf, y_train)
#predicted = random_search.predict(X_test_pca)
predicted = random_search.predict(X_test_tfidf)

print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))

print("F-2 scores: {}".format(metrics.fbeta_score(y_test, predicted, average=None, beta=2)))

print("Confusion matrix: \n{}".format(metrics.confusion_matrix(y_test, predicted)))

print("Now with pipeline: ===========================")


text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=42, max_iter=1000, tol=None, shuffle=True)),
])


text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)


"""
boolean_predicted = predicted.copy()
boolean_test = y_test.copy()
boolean_predicted[boolean_predicted > 0] = 1
boolean_test[boolean_test > 0] = 1
boolean_accuracy = np.mean(boolean_predicted == boolean_test)

print("Boolean clf accuracy: {}".format(boolean_accuracy))
print("Boolean f-2 scores: {}".format(metrics.fbeta_score(boolean_test, boolean_predicted, average=None, beta=2)))
"""

print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))

print("F-2 scores: {}".format(metrics.fbeta_score(y_test, predicted, average=None, beta=2)))

print("Confusion matrix: \n{}".format(metrics.confusion_matrix(y_test, predicted)))



