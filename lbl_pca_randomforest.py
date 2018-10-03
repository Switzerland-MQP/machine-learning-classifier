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


kf = KFold(n_splits=5, shuffle=True)
print("---Loading Data---")
documents = utils.load_dirs_custom([
    './SENSITIVE_DATA/html-tagged',
    './PERSONAL_DATA/html-tagged',
    './NON_PERSONAL_DATA'
])

print("---Creating N_grams---")
documents = utils.n_gram_documents_range(documents, 2, 2)


doc_data, doc_vault, = utils.document_test_train_split(
    documents, 0.10
)

doc_data = np.array(doc_data)

argument_sets = []
for train_index, test_index in kf.split(doc_data):
    print("TRAIN:", train_index, "TEST:", test_index)
    doc_train = doc_data[train_index]
    doc_test = doc_data[test_index]

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
    lbl_scores = []
    document_scores = []
    models = []
    first = True
    for s in argument_sets:
        (X_train, X_test, y_train, y_test) = s
        print("---Fitting model---")
        random_search.fit(X_train, y_train)
        if first is True:
            random_search = random_search.best_estimator_
            first = False

        predicted = random_search.predict(X_test)
        accuracy = fbeta_score(
            y_test,
            predicted,
            average=None,
            beta=2
        )

        print("Line by Line ")
        print("Confusion Matrix: \n{}".format(
            confusion_matrix(y_test, predicted)
        ))

        lbl_scores += [accuracy]
        #  models += [random_search.best_estimator_]
        print("Accuracy: {}".format(accuracy))

    print("lbl Scores: ", lbl_scores)
    print("document Scores: ", document_scores)

    return lbl_scores, models


#  utils.label_new_document("./testFile.txt", random_search)


lbl_scores, models = run_argument_sets(random_search, argument_sets)

average_lbl_scores = [0, 0, 0]
count = 0

for s in lbl_scores:
    average_lbl_scores[0] += s[0]
    average_lbl_scores[1] += s[1]
    average_lbl_scores[2] += s[2]
    count += 1

average_lbl_scores = [x / count for x in average_lbl_scores]


print("Average lbl Scores: ", lbl_scores)
