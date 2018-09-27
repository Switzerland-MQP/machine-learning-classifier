import numpy as np
import xgboost as xgb
import warnings
from scipy.stats import randint as sp_randint
from sklearn.decomposition import TruncatedSVD

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

documents = utils.load_dirs_custom([
    './SENSITIVE_DATA/html-tagged',
    './PERSONAL_DATA/html-tagged',
    './NON_PERSONAL_DATA'
])

doc_train, doc_test, = utils.document_test_train_split(
    documents, 0.20
)

print("Doc train: ", len(doc_train))
print("Doc test: ", len(doc_test))
X_train, y_train = utils.convert_docs_to_lines(doc_train)
X_test, y_test = utils.convert_docs_to_lines(doc_test)

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

print("PCA with XGBoost")

accuracy = fbeta_score(y_test, predicted, average=None, beta=2)
print("Accuracy: {}".format(accuracy))


confusion_matrix(y_test, predicted)

print("Confusion Matrix: \n{}".format(confusion_matrix(y_test, predicted)))


documents_predicted = []
documents_target = []
for doc in doc_test:
    document_count = count_vect.transform(doc.data)
    document_tfidf = tfidf_transformer.transform(document_count)
    document_pca = pca.transform(document_tfidf)

    predicted_lines = gsearch1.predict(document_pca)
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
