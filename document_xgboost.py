import numpy as np
import xgboost as xgb
import warnings
from scipy.stats import randint as sp_randint
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import load_files

# Models to try
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
#  from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.pipeline import Pipeline
import utils

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


resumes = load_files('TEXTDATA/', shuffle=True)
print("Loaded docs")

# Split remainder into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    resumes.data, resumes.target, test_size=0.20
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

# run randomized search
# Accuracy should be comparable to grid search, but runs much much faster
print("Training Model")
n_iter_search = 20

gsearch1 = xgb.XGBClassifier(
    max_depth=10, 
    min_child_weight=4, 
    gamma=0.1, 
    subsample=0.9, 
    colsample_bytree=0.8, 
    scale_pos_weight=1,
    learning_rate=0.1,
    n_estimators=1000,
    seed=27
)

gsearch1.fit(X_train_pca, y_train)
predicted = gsearch1.predict(X_test_pca)

accuracy = fbeta_score(y_test, predicted, average=None, beta=2)
print("Accuracy: {}".format(accuracy))


confusion_matrix(y_test, predicted)

print("Confusion Matrix: \n{}".format(confusion_matrix(y_test, predicted)))
