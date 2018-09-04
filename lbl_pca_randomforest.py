import numpy as np


from scipy.stats import randint as sp_randint
from sklearn.decomposition import TruncatedSVD

# Models to try
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
#  from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score

import utils


data, target, documents = utils.load_dirs_custom([
    './TEXTDATA/SENSITIVE_DATA/html-tagged',
    './TEXTDATA/PERSONAL_DATA/html-tagged',
    './TEXTDATA/NON_PERSONAL_DATA'
])

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.20
)


count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

pca = TruncatedSVD(n_components=200)
pca.fit(X_train_tfidf)

X_train_pca = pca.transform(X_train_tfidf)


# Test data transformations
X_test_count = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_count)
X_test_pca = pca.transform(X_test_tfidf)


clf = RandomForestClassifier()

parameters_rand = {
    "n_estimators": [799],
    "max_depth": [2, 3, 5, 7, None],
    "max_features": [4],
    "min_samples_split": [10],
    "min_samples_leaf": [2],
    "class_weight": ["balanced", {0: 1, 1: 2, 2: 3}, {0: 1, 1: 3, 2: 5},
                     {0: 1, 1: 1, 2: 2}, {0: 1, 1: 1.5, 2: 1.75}],
    "bootstrap": [False],
    "criterion": ["entropy"]
}

# run randomized search
# Accuracy should be comparable to grid search, but runs much much faster
print("Training Model")
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=parameters_rand,
                                   n_iter=n_iter_search,
                                   n_jobs=-1)

random_search.fit(X_train_pca, y_train)
predicted = random_search.predict(X_test_pca)

print("PCA with random forest")

accuracy = fbeta_score(y_test, predicted, average=None, beta=2)
print("Accuracy: {}".format(accuracy))


confusion_matrix(y_test, predicted)

print("Confusion Matrix: \n{}".format(confusion_matrix(y_test, predicted)))


documents_predicted = []
documents_target = []
for doc in documents:
    document_count = count_vect.transform(doc.data)
    document_tfidf = tfidf_transformer.transform(document_count)
    document_pca = pca.transform(document_tfidf)

    predicted_lines = random_search.predict(document_pca)
    predicted_doc = utils.classify_doc(predicted_lines)
    documents_predicted.append(predicted_doc)
    documents_target.append(doc.category)


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
