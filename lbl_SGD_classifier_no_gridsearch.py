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
from sklearn.model_selection import KFold
import utils
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

kfold = KFold(n_splits=5, shuffle=True)

documents = utils.load_dirs_custom([
    './SENSITIVE_DATA/html-tagged',
    './PERSONAL_DATA/html-tagged',
    './NON_PERSONAL_DATA'
])

documents = utils.n_gram_documents_range(documents, 5, 6)
documents = np.array(documents)
doc_train, doc_test, = utils.document_test_train_split(
    documents, 0.20
)

argument_sets = []
for train_index, test_index in kfold.split(documents):
    print("TRAIN:", train_index, "TEST:", test_index)
    doc_train, doc_test = documents[train_index], documents[test_index]
    X_train, y_train = utils.convert_docs_to_lines(doc_train)
    X_test, y_test = utils.convert_docs_to_lines(doc_test)
    preprocessing = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                    ('tfidf', TfidfTransformer())])
    preprocessing.fit(X_train)
    X_train, X_test = (preprocessing.transform(X_train), preprocessing.train(X_test))
    argument_sets += [(X_train, X_test, y_train, y_test)] 




text_clf = SGDClassifier(loss='hinge', penalty='none', learning_rate='optimal', alpha=1e-4, epsilon=0.1, max_iter=1000, tol=None, shuffle=True)

def run_argument_sets(text_clf, argument_sets):
    scores = []
    for s in argument_sets:
        (X_train, X_test, y_train, y_test) = s
        print("---Fitting model---")
        text_clf.fit(X_train, y_train)

        print("SVM with SGD")
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
      
        scores += [text_clf.score(X_test, y_test)]
        
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
    print("Scores: ", scores)
    print("Scores:", np.mean(scores))

run_argument_sets(text_clf, argument_sets)
