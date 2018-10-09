import time
start = time.time()

from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from sklearn.metrics import fbeta_score, confusion_matrix, recall_score 


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
    '../../TAGGED_DATA_NEW_NEW/SENSITIVE_DATA/html-tagged',
    '../../TAGGED_DATA_NEW_NEW/PERSONAL_DATA/html-tagged',
    '../../TAGGED_DATA_NEW_NEW/NON_PERSONAL_DATA'
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
y = y_encoded



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






cutoff_dict = {}
for i in range(22):
	category = utils.all_categories_dict[i+1]
	predicted_category = predicted[:,i+1]
	y_category = y_test[:,i+1]

	cutoff = cutoff_graph(predicted_category, y_category)
	cutoff_dict[category] = cutoff

	predicted_category = np.where(predicted_category > cutoff, 1, 0)
	#score = fbeta_score(predicted_category, y_category, average=None, beta=3)
	score = np.mean(y_category == predicted_category)
	f2 = fbeta_score(y_category, predicted_category, average=None, beta=2)
	print(f"Accuracy for category: {category} : {score} -- f2: {f2}")
	print(confusion_matrix(y_category, predicted_category))	

f = open("./category-clf/cutoffs.pickle", "wb")
f.write(pickle.dumps(cutoff_dict))
f.close()
print("Finished dumping cutoff pickle")



elapsed = time.time() - start
print("Elapsed time:", elapsed)
print(f"Stopped at epoch {stopped}")

def print_results(predicted, y_test):
	f2_scores = fbeta_score(y_test, predicted, average=None, beta=2)

	print("F-2 scores: {}  | Average: {}".format(f2_scores, np.mean(f2_scores)))

	print("Confusion matrix: \n{}".format(confusion_matrix(y_test, predicted)))

#print_results(predicted_vec, y_test)

def show_overfit_plot():
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.legend(['train','test'], loc='upper left')
	plt.show()

#show_overfit_plot()



### Save model configuration and weights ###
def save():
	model_json = nn.to_json()
	with open("./category-clf/model.json", "w") as json_file:
		json_file.write(model_json)
	json_file.close()
	nn.save_weights("./category-clf/model.h5")

save()

