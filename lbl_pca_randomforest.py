import numpy as np

import os

from scipy.stats import randint as sp_randint
from sklearn.datasets import load_files
from sklearn.decomposition import TruncatedSVD

# Models to try
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
#  from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import fbeta_score

from parser.run_parser import run_parser 

personal_categories = ['name','id-number', 'location', 'online-id', 'dob', 'phone', 'physical', 'physiological', 'professional', 'genetic', 'mental', 'economic', 'cultural', 'social']
sensitive_categories = ['criminal', 'origin', 'health', 'religion', 'political', 'philosophical', 'unions', 'sex-life', 'sex-orientation', 'biometric']


def convert_categories(categories):
    for c in sensitive_categories:
        if c in categories:
            return 2
    for c in personal_categories:
        if c in categories:
            return 1
    return 0


def read_dir():
    documents = {}
    all_lines = []
    for root, dirs, files in os.walk("./TEXTDATA"):
        for file in files:
            full_path = root + '/' + file
            lines = run_parser(full_path)
            documents[full_path] = lines
            all_lines = all_lines + lines
    print(all_lines)
    return all_lines


lines = read_dir()

data = np.array([])
target = np.array([])
for l in lines:
    data = np.append(data, [l.text])
    target = np.append(target, [convert_categories(l.categories)])


X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.20
)


count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

pca = TruncatedSVD(n_components=20)
pca.fit(X_train_tfidf)

X_train_pca = pca.transform(X_train_tfidf)


# Test data transformations
X_test_count = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_count)
X_test_pca = pca.transform(X_test_tfidf)


clf = RandomForestClassifier()

parameters_rand = {
    "n_estimators": sp_randint(300, 2000),
    "max_depth": [3, None],
    "max_features": sp_randint(1, 11),
    "min_samples_split": sp_randint(2, 11),
    "min_samples_leaf": sp_randint(1, 11),
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"]
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
