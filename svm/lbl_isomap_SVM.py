import numpy as np


#  from scipy.stats import randint as sp_randint
from sklearn.decomposition import TruncatedSVD
from scipy.stats import randint as sp_randint

# Models to try
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#  from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import utils
from sklearn.svm import SVC

documents = utils.load_dirs_custom([
    './SENSITIVE_DATA/html-tagged',
    './PERSONAL_DATA/html-tagged',
    './NON_PERSONAL_DATA'
])

doc_train, doc_test, = utils.document_test_train_split(
    documents, 0.4
)

print("Doc train: ", len(doc_train))
print("Doc test: ", len(doc_test))

X_train, y_train = utils.convert_docs_to_lines(doc_train)
X_test, y_test = utils.convert_docs_to_lines(doc_test)

order = np.arange(len(X_train))
np.random.shuffle(order)

n = 10000

X_train, y_train = (X_train[order][:n], y_train[order][:n])



'''
vect = CountVectorizer()
X_train_count = vect.fit_transform(X_train)

tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_count)
pca = TruncatedSVD(n_components=20)
X_train_pca = pca.fit_transform(X_train_tfidf)
isomap = LocallyLinearEmbedding(n_neighbors=5, n_components=2)
LocallyLinearEmbedding.fit(X_train_pca)
X_train_isomap = LocallyLinearEmbedding.transform(X_train_pca)

X_test_count = vect.transform(X_test)

X_test_tfidf = tfidf.transform(X_test_count)
X_test_pca = pca.fit(X_test_tfidf)
X_test_isomap = isomap.transform(X_test_pca)


clf = SGDClassifier(loss='hinge', penalty='none', learning_rate='optimal', alpha=1e-4, epsilon=0.1, max_iter=1000, tol=None, shuffle=True)
'''
clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('pca', TruncatedSVD(n_components=200)),
                     ('iso', Isomap(n_neighbors=5, n_components=20,)),
                     ('clf', SVC(kernel='linear', C=1.0, gamma=0.1))
                     ])

clf.fit(X_train, y_train)


print("PCA with Isomaps RBF Kernel SVM")

documents_predicted = []
documents_target = []
all_predicted_lines = []
all_target_lines = []
for doc in doc_test:
    predicted_lines = clf.predict(doc.data)
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

