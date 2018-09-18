import time
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, confusion_matrix 

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.regularizers import l1
from keras import backend as K
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt



### Begin Aux Functions ###

def show_overfit_plot():
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.legend(['train','test'], loc='upper left')
	plt.show()

def show_variance_plot():
	explained = preprocessing.named_steps['pca'].explained_variance_
	cumulative = [np.sum(explained[:i]) for i in range(len(explained))]
	#plt.plot(explained)
	plt.plot(cumulative)
	plt.legend(['explained variance','cumulative explained variance'], loc='upper left')
	plt.show()

def to_1_interval(arr):
	minimum = min(arr)
	maximum = max(arr)
	new_arr = []
	for value in arr:
		new_arr.append((value-minimum)/(maximum-minimum))
	return new_arr

def smooth(keep, arr):
	smoothed = []
	previous = arr[0]
	for i in range(len(arr)):
		previous = keep*previous + (1-keep)*arr[i]
		smoothed.append(previous)
	return smoothed

def print_results(predicted, y_test):
	print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))
	
	f2_scores = fbeta_score(y_test, predicted, average=None, beta=2)
	print("F-2 scores: {}".format(f2_scores))
	print(confusion_matrix(y_test, predicted))


### Begin main parameterized code ###

def run_model(layers):
	print("==============================================")
	start = time.time()

	documents = load_files('../TEXTDATA/', shuffle=False)
	x_train, x_test, y_train, y_test = train_test_split(
	    documents.data, documents.target, test_size=0.3
	)

	x_test_copy = x_test.copy()

	preprocessing = Pipeline([('count', CountVectorizer()),
												  ('tfidf', TfidfTransformer()),
													('pca', TruncatedSVD(n_components=430))])
	preprocessing.fit(x_train)
	x_train, x_test = (preprocessing.transform(x_train), preprocessing.transform(x_test))
	print("Data preprocessing took {} seconds.".format(time.time()-start))


	input_shape = x_train.shape[1]
	nn = Sequential()
	for layer in layers(input_shape):
		nn.add(layer)
	nn.add(Dense(3,  activation='softmax', name="out_layer"))
	nn.compile(loss= 'categorical_crossentropy',
           optimizer='adam',
           metrics=['mean_squared_logarithmic_error'])
	print(nn.get_config())
		
	y_train = np_utils.to_categorical(y_train)
	y_test_onehot = np_utils.to_categorical(y_test)


	def fit(batch_size, epochs):
		return nn.fit(x_train, y_train,
								batch_size=batch_size,
								epochs=epochs,
								verbose=0,
								validation_data=(x_test, y_test_onehot))


	history = fit(128, 500)
	predicted_vec = nn.predict(x_test)
	predicted = np.argmax(predicted_vec, axis=1)

	print("Elapsed time:", time.time()-start)
	print_results(predicted, y_test)
	return np.mean(fbeta_score(y_test, predicted, average=None, beta=2))


def run_model_average(layers, n_runs):
	f2_scores = [run_model(layers) for i in range(n_runs)]

layers = ((lambda input_shape: [
	Dense(16, activation='relu', input_shape=(input_shape,)),
	Dropout(0.25),
]))

f2_score = run_model(layers)
print(f"F-2 score: {f2_score}")

	















