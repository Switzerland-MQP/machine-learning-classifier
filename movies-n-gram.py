import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import randint as sp_randint
from sklearn.datasets import load_files
from sklearn.decomposition import PCA

# Models to try
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import Data
# Downloaded from http://www.nltk.org/nltk_data/
moviedir = 'data/movie_reviews/txt_sentoken/'
reviews = load_files(moviedir, shuffle=True)

# Hide some data in the vault to avoid snooping
x_work, x_vault, y_work, y_vault = train_test_split(
    reviews.data, reviews.target, test_size=0.10
)

# Split remainder into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    x_work, y_work, test_size=0.10
)

params = {'bootstrap': False,
    'criterion': 'entropy',
    'max_depth': 3,
    'max_features': 5,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 457}

text_clf = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,3))),
                    ('svd', TruncatedSVD(n_components=120)),
                    ('clf', RandomForestClassifier(**params)),
])

text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)

print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))
