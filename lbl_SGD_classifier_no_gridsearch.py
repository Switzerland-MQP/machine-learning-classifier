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

documents = utils.n_gram_documents(documents, 2)

doc_train, doc_test, = utils.document_test_train_split(
    documents, 0.20
)

print("Doc train: ", len(doc_train))
print("Doc test: ", len(doc_test))
X_train, y_train = utils.convert_docs_to_lines(doc_train)
X_test, y_test = utils.convert_docs_to_lines(doc_test)




text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                    ('tfidf', TfidfTransformer()),
										
                    ('clf', SGDClassifier(loss='hinge', penalty='none', learning_rate='optimal', alpha=1e-4, epsilon=0.1, max_iter=1000, tol=None, shuffle=True)),
])

print("Training Model")
text_clf.fit(X_train, y_train)
print("PCA with SGD")

documents_predicted = []
documents_target = []
all_predicted_lines = []
all_target_lines = []
for doc in doc_test:
    predicted_lines = text_clf.predict(doc.data)
    all_predicted_lines += list(predicted_lines)
    all_target_lines += list(doc.targets)

    predicted_doc = utils.classify_doc(predicted_lines)
    documents_predicted.append(predicted_doc)
    documents_target.append(doc.category)


print("Line by Line ")
total_accuracy = np.mean(predicted_lines == doc.category)
print("Total Accuracy:")
print(total_accuracy)

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
total_doc_accuracy = np.mean(predicted_doc == doc.targets)
print("Total Accuracy:")
print(total_doc_accuracy)
print("Document Confusion Matrix: \n{}".format(
    confusion_matrix(documents_target, documents_predicted)
))
