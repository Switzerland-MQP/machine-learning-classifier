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
import utils

from keras.utils import np_utils

def load_keras_model(json_path, weights_path):
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
    return loaded_model

preprocessing = pickle.load(open("./line-clf/preprocessing.pickle", "rb"))
clf = load_keras_model("./line-clf/model.json",
											"./line-clf/model.h5")

documents = utils.load_dirs_custom([
    '../../vault_line/sensitive',
    '../../vault_line/personal',
    '../../vault_line/nonpersonal'
])
documents = utils.n_gram_documents_range(documents, 8, 8)
x, y = utils.convert_docs_to_lines(documents)

x = preprocessing.transform(x)

y_predicted = np.where(clf.predict(x) >= 0.5, 1, 0)
y = np.where(y == 0, 0, 1)



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

#  import ipdb
#  ipdb.set_trace()

print_results(y_predicted, y)




