import matplotlib.pyplot as plt
import numpy as np
import random

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


documents = load_files('../TEXTDATA/', shuffle=False)


data = np.load('autoencoded_data.npz')
x = data['x']
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30
)

#import code
#code.interact(local=locals())

text_clf = Pipeline([('clf', SGDClassifier(loss='hinge', penalty='none', learning_rate='optimal', alpha=1e-4, epsilon=0.1, max_iter=1000, tol=None, shuffle=True)),
])


text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)

boolean_predicted = predicted.copy()
boolean_test = y_test.copy()
boolean_predicted[boolean_predicted > 0] = 1
boolean_test[boolean_test > 0] = 1
boolean_accuracy = np.mean(boolean_predicted == boolean_test)

print("Boolean clf accuracy: {}".format(boolean_accuracy))
print("Boolean f-2 scores: {}".format(metrics.fbeta_score(boolean_test, boolean_predicted, average=None, beta=2)))

print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))

print("F-2 scores: {}".format(metrics.fbeta_score(y_test, predicted, average=None, beta=2)))

print("Confusion matrix: \n{}".format(metrics.confusion_matrix(y_test, predicted)))



