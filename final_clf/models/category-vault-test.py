import time
start = time.time()

from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from sklearn.metrics import fbeta_score, confusion_matrix, recall_score 

from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.regularizers import l1
from keras.callbacks import EarlyStopping
import utils

from keras import backend as K

import numpy as np
import pickle

from keras.utils import np_utils



documents = utils.load_dirs_custom([
    '../../vault_line/sensitive',
    '../../vault_line/personal',
    '../../vault_line/nonpersonal'
], individual=True)

x = []
y = []
for document in documents:
	lines = document.lines
	categories = []
	for line in lines:
		for category in line.categories:
			if category not in categories:
				categories.append(category)
	x += ['\n'.join(document.data)]
	y += [categories]

y_encoded = []
for categories in y:
	one_hot_encoded = np.zeros(23)
	for category in categories:
		i = utils.all_categories_dict.inv[category]
		one_hot_encoded[i] = 1
	y_encoded += [one_hot_encoded]
y = np.array(y_encoded)



def load_keras_model(json_path, weights_path):
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
    return loaded_model

preprocessing = pickle.load(open("./category-clf/preprocessing.pickle", "rb"))
clf = load_keras_model("./category-clf/model.json",
											"./category-clf/model.h5")
x = preprocessing.transform(x)

predicted = clf.predict(x)






cutoffs = pickle.load(open("./category-clf/cutoffs.pickle", "rb"))
for i in range(22):
	category = utils.all_categories_dict[i+1]
	predicted_category = predicted[:,i+1]
	y_category = y[:,i+1]

	cutoff = cutoffs[category]

	predicted_category = np.where(predicted_category > cutoff, 1, 0)
	#score = fbeta_score(predicted_category, y_category, average=None, beta=3)
	score = np.mean(y_category == predicted_category)
	f2 = fbeta_score(y_category, predicted_category, average=None, beta=2)
	print(f"Accuracy for category: {category} : {score} -- f2: {f2}")
	print(confusion_matrix(y_category, predicted_category))	


