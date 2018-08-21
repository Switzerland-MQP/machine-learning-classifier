import matplotlib.pyplot as plt
import numpy as np

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

reviews = load_files('TEXTDATA/', shuffle=True)

# Hide some data in the vault to avoid snooping
x_work, x_vault, y_work, y_vault = train_test_split(
    reviews.data, reviews.target, test_size=0.10
)

# Split remainder into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    x_work, y_work, test_size=0.10
)

text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=42, max_iter=10, tol=None)),
])


text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)

print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))





