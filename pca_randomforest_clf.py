import numpy as np

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
print("Accuracy: {}".format(np.mean(predicted == y_test)))


#  tn, fp , fn, tp = metrics.confusion_matrix(y_test, predicted).ravel()
#  print("True negatives:{0}\nFalse Positives:{1}\nFalse Negatives:{2}\nTrue Positives:{3}".format(tn, fp, fn, tp))
#  print("F1 score:{}".format(metrics.f1_score(y_test, predicted)))

#  false_positives = []
#  true_positives = []
#  for predicted_label, actual_label, example in zip(predicted, y_test, X_test):
    #  # print("{0}, {1}".format(predicted_label, actual_label))
    #  if predicted_label == 1 and actual_label == 0:
        #  # print("Found one!")
        #  false_positives += [example]
    #  elif predicted_label == 1 and actual_label == 1:
        #  true_positives += [example]
#  print("False positives:", len(false_positives))
#  print(false_positives[0])
#  print("\n")
#  print(true_positives[0])


confusion_matrix(y_test, predicted)

print("Confusion Matrix: \n{}".format(confusion_matrix(y_test, predicted)))
