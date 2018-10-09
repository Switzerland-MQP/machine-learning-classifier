import time
start = time.time()

from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from sklearn.metrics import fbeta_score, confusion_matrix 

from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.regularizers import l1
from keras.callbacks import EarlyStopping

from keras import backend as K

import numpy as np
import pickle

from keras.utils import np_utils

def load_keras_model(json_path, weights_path):
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
    return loaded_model

preprocessing = pickle.load(open("./document-clf/preprocessing.pickle", "rb"))
clf = load_keras_model("./document-clf/model.json",
											"./document-clf/model.h5")

documents = utils.load_dirs_custom([
    '../../SUBSET/SENSITIVE_DATA/html-tagged',
    '../../SUBSET/PERSONAL_DATA/html-tagged',
    '../../SUBSET/NON_PERSONAL_DATA'
])

x = []
y_categories = []
for document in documents:
	lines = document.lines
	categories = []
	for line in lines:
		for category in line.categories:
			if category not in categories:
				categories.append(category)
	x += ['\n'.join(document.data)]
	y_categories += [categories]

personal_categories = utils.personal_categories_dict.values()
sensitive_categories = utils.sensitive_cateogires_dict.values()

y = []
for categories in y_categories:
	




def recall(y_true, y_pred):
	matrix = confusion_matrix(y_true, y_pred)
	scores = []
	for i in range(len(matrix)):
			row = matrix[i]
			correct = row[i]
			total = sum(row)
			scores.append(correct/total)
	return scores 





def print_results(predicted, y):
	print("Classifier accuracy: {}".format(np.mean(predicted == y)))
	f2_scores = fbeta_score(y, predicted, average=None, beta=2)

	print(f"F-2 scores: {f2_scores}")
	print("Confusion matrix: \n{}".format(confusion_matrix(y, predicted)))
	print(f"Recalls: {recall(y, predicted)}")


print_results(y_predicted, y)




