import numpy as np


from scipy.stats import randint as sp_randint
from sklearn.decomposition import TruncatedSVD

# Models to try
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GridSearchCV
import utils
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

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




clf = SGDClassifier(shuffle=True, tol=None, max_iter=1000)

params = {
 "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron", "squared_loss", "huber"],
 "penalty": ['l1', 'l2', 'none'],
 "class_weight": [{0:1, 1:1.5, 2:1.75}, {0:1, 1:2, 0:3}, {0:1, 1:3, 0:5},"balanced", None],
 "learning_rate": ['optimal']
}

gridsearch_clf = Pipeline([
	('vect', CountVectorizer()),
  ('tfidf', TfidfTransformer()),
	('gridsearch', GridSearchCV(clf, param_grid=params, n_jobs=-1))						
])

print("Training Model")
gridsearch_clf.fit(X_train, y_train)
print("PCA with random forest")

documents_predicted = []
documents_target = []
all_predicted_lines = []
all_target_lines = []
for doc in doc_test:
    predicted_lines = gridsearch_clf.predict(doc.data)
    all_predicted_lines += list(predicted_lines)
    all_target_lines += list(doc.targets)

    predicted_doc = utils.classify_doc(predicted_lines)
    documents_predicted.append(predicted_doc)
    documents_target.append(doc.category)


print("Line by Line ")
print("Confusion Matrix: \n{}".format(
    confusion_matrix(all_target_lines, all_predicted_lines)
))

accuracy = fbeta_score(
    all_target_lines,
    all_predicted_lines,
    average=None,
    beta=2
)
print("Accuracy: {}".format(accuracy))


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
