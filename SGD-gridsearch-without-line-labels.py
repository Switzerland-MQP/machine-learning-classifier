import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import GridSearchCV

from scipy.stats import randint as sp_randint
from sklearn.datasets import load_files
from sklearn.decomposition import PCA

# Models to try
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier

from sklearn import metrics

# Import Data
# Downloaded from http://www.nltk.org/nltk_data/

documents = load_files('TEXTDATA/', shuffle=True)

# Split remainder into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    documents.data, documents.target, test_size=0.40
)


clf = SGDClassifier(shuffle=True, tol=None, max_iter=1000)
params = {
 "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron", "squared_loss", "huber"],
 "penalty": ['l1', 'l2', 'none']
}

gridsearch_clf = Pipeline([
	('vect', CountVectorizer()),
  ('tfidf', TfidfTransformer()),
	('gridsearch', GridSearchCV(clf, param_grid=params, n_jobs=-1))						
])


gridsearch_clf.fit(X_train, y_train)
predicted = gridsearch_clf.predict(X_test)

print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))

print("F-2 scores: {}".format(metrics.fbeta_score(y_test, predicted, average=None, beta=2)))

print("Confusion matrix: \n{}".format(metrics.confusion_matrix(y_test, predicted)))

boolean_predicted = predicted.copy()
boolean_test = y_test.copy()
boolean_predicted[boolean_predicted > 0] = 1
boolean_test[boolean_test > 0] = 1
boolean_accuracy = np.mean(boolean_predicted == boolean_test)

print("Boolean clf accuracy: {}".format(boolean_accuracy))
print("Boolean f-2 scores: {}".format(metrics.fbeta_score(boolean_test, boolean_predicted, average=None, beta=2)))


print("Now with no grid search: ===========================")







text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=42, max_iter=1000, tol=None, shuffle=True)),
])


text_clf.fit(X_train, y_train)
predicted2 = text_clf.predict(X_test)


boolean_predicted = predicted2.copy()
boolean_test = y_test.copy()
boolean_predicted[boolean_predicted > 0] = 1
boolean_test[boolean_test > 0] = 1
boolean_accuracy = np.mean(boolean_predicted == boolean_test)

print("Boolean clf accuracy: {}".format(boolean_accuracy))
print("Boolean f-2 scores: {}".format(metrics.fbeta_score(boolean_test, boolean_predicted, average=None, beta=2)))

print("Classifier accuracy: {}".format(np.mean(predicted2 == y_test)))

print("F-2 scores: {}".format(metrics.fbeta_score(y_test, predicted2, average=None, beta=2)))

print("Confusion matrix: \n{}".format(metrics.confusion_matrix(y_test, predicted2)))



