import time
import random
import os
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, confusion_matrix 

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.regularizers import l1
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

begin = time.time()

### Begin Aux Functions ###

def print_model_summary(model):
	config = model.get_config()
	print("====== Model Summary ======")
	_ = ' '
	for item in config:
		name = item['class_name']
		if name == 'Dense':
			units = item['config']['units']
			activation = item['config']['activation']
			print(f"{name}\t\t{units}\t\t{activation}")
		else:
			rate = item['config']['rate']
			print(f"{name}\t\tRate:{rate}")

def print_results(predicted, y_test):
	print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))
	
	f2_scores = fbeta_score(y_test, predicted, average=None, beta=2)
	print("F-2 scores: {} - average: {}".format(f2_scores, np.mean(f2_scores)))
	print(confusion_matrix(y_test, predicted))

def show_overfit_plot(history):
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

### Begin main parameterized code ###

def run_model(nn, x_train, x_test, y_train, y_test):
	start = time.time()


	x_test_copy = x_test.copy()

	input_shape = x_train.shape[1]

	y_test_onehot = np_utils.to_categorical(y_test)	

	early_stopping_monitor = EarlyStopping(monitor='val_loss', 
																			min_delta=0, 
																			patience=10, 
																			verbose=0, mode='auto')
	def fit(batch_size, epochs):
		return nn.fit(x_train, y_train,
								batch_size=batch_size,
								epochs=epochs,
								verbose=0,
								validation_data=(x_test, y_test_onehot),
								callbacks= [early_stopping_monitor])


	history = fit(128, 500)
	predicted_vec = nn.predict(x_test)
	
	"""
	predicted = []
	for probabilities in predicted_vec:
		if probabilities[2] > 0.25:
			predicted += [2]
			continue
		if probabilities[1] > 0.25:
			predicted += [1]
			continue
		predicted += [0]
	"""
	predicted = np.argmax(predicted_vec, axis=1)

	stopped_epoch = early_stopping_monitor.stopped_epoch
	#print(f"Stopped training at epoch {stopped_epoch}")
	#print(f"Elapsed time: {time.time()-begin}")
	#print_results(predicted, y_test)
	scores = fbeta_score(y_test, predicted, average=None, beta=2)
	mean = np.mean(scores)
	print(f"Elapsed time: {time.time()-begin} | Epochs: {stopped_epoch} | F2-score: {mean}")
	print(confusion_matrix(y_test, predicted))
	return (mean, stopped_epoch, scores)


#def run_model_average(layers):
#	n_runs = 3
#	f2_scores = [run_model(layers) for i in range(n_runs)]
#	average = np.mean(f2_scores)
#	print("Total elapsed: {}".format(time.time()-begin))
#	print(f"####f2-score, average of {n_runs} runs: {average}####")
#	return average

def run_argument_sets(nn, argument_sets):
	print("==========Running 5 fold cross validation==========")
	results = []
	epochs = []
	scores_ = []
	for argument_set in argument_sets:
		result, epoch, scores = run_model(nn, *argument_set)
		results += [result]
		epochs += [epoch]
		scores_ += [scores]
	avgs = np.mean(scores_, 0)
	print(f"Avgs: {avgs} ")

	print(f"Model average F2: {np.mean(results)}\n Average epochs:{np.mean(epochs)}")
	return results
	
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
	

def run_model_kfold(nn):
	f = "5-cv-preprocessed-data.npy"
	if not os.path.isfile(f):
		print("No saved data found, creating it----")
		create_and_save_folds(5, f)
	
	argument_sets = np.load(f)
	print("Data preprocessing took {} seconds.".format(time.time()-begin))
	results = run_argument_sets(nn, argument_sets)
	return np.mean(results)

from functools import reduce	
def create_random_nn():
	nn = Sequential()
	nn.add(Dense(random.randint(32, 600), activation="relu", input_shape=(430,)))
	nn.add(Dropout(random.uniform(0, 0.7)))
	for i in range(random.choice([0, 1, 2, 3])):
		nn.add(Dense(random.randint(32, 600), activation="relu"))
		nn.add(Dropout(random.uniform(0, 0.7)))
	nn.add(Dense(3,  activation='softmax', name="out_layer"))
	nn.compile(loss= 'categorical_crossentropy',
           optimizer='adam',
           metrics=['mean_squared_logarithmic_error'])
	return nn

"""nns = []
for i in range(180):
	nn = create_random_nn()
	print_model_summary(nn)
	result = run_model_kfold(nn)
	nns += [(result, nn)]
	print("||||||||||||||||||||| done with ", i)

nns = sorted(nns, key=(lambda x: x[0]), reverse=True)
print(f"Best model score: {nns[0][0]}")
print_model_summary(nns[0][1])
"""
def create_nn():
	nn = Sequential()
	nn.add(Dense(128, activation="relu", input_shape=(430,)))
	nn.add(Dropout(0.25))
	nn.add(Dense(32, activation="relu"))
	nn.add(Dropout(0.25))
	nn.add(Dense(3,  activation='sigmoid', name="out_layer"))
	nn.compile(loss= 'categorical_crossentropy',
           optimizer='adam',
           metrics=['mean_squared_logarithmic_error'])
	return nn



result =	[run_model_kfold(
	create_nn()
)]	

print(result)
