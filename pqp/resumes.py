# Trying to train a model to distinguish movie industry resumes from
# movie reviews

import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

resumedir = 'data/resumes_and_reviews'
resumes = load_files(resumedir, shuffle=True)

# Split remainder into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    resumes.data, resumes.target, test_size=0.50
)

count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)
resume_matrix = X_train_tfidf[y_train == 0]
top_words = []

svd = TruncatedSVD(n_components=20)
svd.fit(X_train_tfidf)
# print("svd explained variance: {}".format(svd.explained_variance_.sum()))
X_train_svd = svd.transform(X_train_tfidf)

# Test data transformations
X_test_count = count_vect.transform(X_test)
X_tfidf_test = tfidf_transformer.transform(X_test_count)
X_test_svd = svd.transform(X_tfidf_test)

params = {
    'bootstrap': False,
    'criterion': 'entropy',
    'max_depth': 3,
    'max_features': 5,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 457
}

clf = RandomForestClassifier(**params)
clf.fit(X_train_svd, y_train)
predicted = clf.predict(X_test_svd)

print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))
print("Score: {}".format(clf.score(X_test_svd, y_test)))

confusion_matrix(y_test, predicted)
tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
f1 = f1_score(y_test, predicted)

print("Confusion Matrix: \n{}".format(confusion_matrix(y_test, predicted)))
print("F1 Score: {}".format(f1))
