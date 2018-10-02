import matplotlib.pyplot as plt
import numpy as np
import random
import os, time
begin = time.time()

from scipy.stats import randint as sp_randint
from sklearn.datasets import load_files
from sklearn.decomposition import PCA

# Models to try
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score

from sklearn.linear_model import SGDClassifier

from sklearn import metrics

def run_model(x_train, x_test, y_train, y_test):
	y_train = np.argmax(y_train, 1)

	svm = SGDClassifier(loss='hinge', penalty='none', learning_rate='optimal', alpha=1e-4, epsilon=0.1, max_iter=1000, tol=None, shuffle=True)
	print(f" xtrain shape {x_train.shape}, y_train shape {y_train.shape}")
	svm.fit(x_train, y_train)

	predicted = svm.predict(x_test)
	scores = fbeta_score(y_test, predicted, average=None, beta=2)
	mean = np.mean(scores)
	print(f"Elapsed time: {time.time()-begin} | F2-score: {mean}")
	return scores

def run_argument_sets(argument_sets):
	print("==========Running 5 fold cross validation==========")
	scores_ = []
	for argument_set in argument_sets:
		scores = run_model(*argument_set)
		scores_ += [scores]
	avgs = np.mean(scores_, 0)
	print(f"Avgs: {avgs} ")
	return avgs
	
def create_and_save_folds(k, f):
	kfold = KFold(n_splits=k)
	documents = load_files('../TEXTDATA/', shuffle=True)
	x = np.array(documents.data)
	y = np.array(documents.target)

	argument_sets = []
	print(len(documents.data))
	for train_indices, test_indices in kfold.split(x):
		print(f"TRAIN: {train_indices} | TEST: {test_indices}")
		x_train, x_test = x[train_indices], x[test_indices]
		y_train, y_test = y[train_indices], y[test_indices]
		preprocessing = Pipeline([('count', CountVectorizer()),
												  ('tfidf', TfidfTransformer()),
													('pca', TruncatedSVD(n_components=430))])
		preprocessing.fit(x_train)
		x_train, x_test = (preprocessing.transform(x_train), preprocessing.transform(x_test))	
		y_train = np_utils.to_categorical(y_train)
	
		argument_sets += [(x_train, x_test, y_train, y_test)]
	np.save(f, argument_sets)
	

def run_model_kfold():
	f = "5-cv-preprocessed-data.npy"
	if not os.path.isfile(f):
		print("No saved data found, creating it----")
		create_and_save_folds(5, f)
	
	argument_sets = np.load(f)
	print("Data preprocessing took {} seconds.".format(time.time()-begin))
	results = run_argument_sets(argument_sets)
	return results

results = run_model_kfold()
print(f"F2-scores: {results}")

