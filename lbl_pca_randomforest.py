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

from sklearn.pipeline import Pipeline
import utils

from sklearn.model_selection import KFold


kf = KFold(n_splits=5)
print("---Loading Data---")
documents = utils.load_dirs_custom([
    './TEXTDATA/SENSITIVE_DATA/html-tagged',
    './TEXTDATA/PERSONAL_DATA/html-tagged',
    './TEXTDATA/NON_PERSONAL_DATA'
])

print("---Creating N_grams---")
documents = utils.n_gram_documents_range(documents, 2, 2)


doc_train, doc_test, = utils.document_test_train_split(
    documents, 0.20
)

argument_sets = []
for train_index, test_index in kf.split(documents):
    print("TRAIN:", train_index, "TEST:", test_index)
    doc_train, doc_test = documents[train_index], documents[test_index]

    X_train, y_train = utils.convert_docs_to_lines(doc_train)
    X_test, y_test = utils.convert_docs_to_lines(doc_test)
    argument_sets += [(X_train, X_test, y_train, y_test)]


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('pca', TruncatedSVD(n_components=20)),
                     ('clf', RandomForestClassifier(n_jobs=-1))
                     ])

param_distributions = {
    "vect__ngram_range": [(1, 3)],
    "pca__n_components": sp_randint(20, 400),
    "clf__n_estimators": sp_randint(100, 2000),
    "clf__max_features": sp_randint(1, 8),
    "clf__min_samples_leaf": sp_randint(1, 6),
    #  "clf__class_weight": [
        #  {0: 1, 1: 1.5, 2: 1.75},
        #  {0: 1, 1: 2, 2: 3},
        #  {0: 1, 1: 3, 2: 5},
    #  ],
    "clf__criterion": ["entropy", "gini"]
}


n_iter_search = 10
random_search = RandomizedSearchCV(
    text_clf,
    param_distributions=param_distributions,
    n_iter=n_iter_search,
    n_jobs=-1
)


def run_argument_sets(random_search, argument_sets):
    scores = []
    for s in argument_sets:
        (X_train, X_test, y_train, y_test) = s
        print("---Fitting model---")
        random_search.fit(X_train, y_train)

        print("PCA with random forest")
        documents_predicted = []
        documents_target = []
        all_predicted_lines = []
        all_target_lines = []
        for doc in doc_test:
            predicted_lines = random_search.predict(doc.data)
            all_predicted_lines += list(predicted_lines)
            all_target_lines += list(doc.targets)

            predicted_doc = utils.classify_doc(predicted_lines)
            documents_predicted.append(predicted_doc)
            documents_target.append(doc.category)
        scores += [random_search.score(X_test, y_test)]

        #  print("Line by Line ")
        #  print("Confusion Matrix: \n{}".format(
            #  confusion_matrix(all_target_lines, all_predicted_lines)
        #  ))

        #  accuracy = fbeta_score(
            #  all_target_lines,
            #  all_predicted_lines,
            #  average=None,
            #  beta=2
        #  )
        #  print("Accuracy: {}".format(accuracy))

        #  doc_accuracy = fbeta_score(
            #  documents_target,
            #  documents_predicted,
            #  average=None,
            #  beta=2
        #  )

        #  print("Document Accuracy: {}".format(doc_accuracy))

        #  print("Document Confusion Matrix: \n{}".format(
            #  confusion_matrix(documents_target, documents_predicted)
        #  ))

    print(f"Scores: {scores}")
    print(f"Score mean: {np.mean(scores)}")


#  utils.label_new_document("./testFile.txt", random_search)


run_argument_sets(random_search, argument_sets)
