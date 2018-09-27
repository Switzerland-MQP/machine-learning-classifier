import time
start = time.time()

from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from sklearn.metrics import fbeta_score, confusion_matrix 


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.regularizers import l1
from keras.callbacks import EarlyStopping

from keras import backend as K

import numpy as np

from keras.utils import np_utils

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
print("Finished data preprocessing - {} elapsed".format(time.time()-start))

early_stopping_callback = EarlyStopping(monitor='val_loss',
				min_delta=0, patience=12, verbose=0, mode='auto')


input_shape = x_train.shape[1]
nn = Sequential()
nn.add(Dense(128, activation='relu', input_shape=(input_shape,)))
nn.add(Dropout(0.25))
nn.add(Dense(32, activation='relu'))
nn.add(Dropout(0.25))
nn.add(Dense(3,  activation='softmax', name="out_layer"))
nn.compile(loss= 'categorical_crossentropy',
           optimizer='adam',
           metrics=['mean_squared_logarithmic_error'])

print("Begin fitting network")

y_train = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)

def fit(batch_size, epochs):
	global x_train, y_train, x_test, y_test_onehot
	return nn.fit(x_train, y_train,
								batch_size=batch_size,
								epochs=epochs,
								verbose=0,
								validation_data=(x_test, y_test_onehot),
								callbacks=[early_stopping_callback])
stopped = early_stopping_callback.stopped_epoch

history = fit(196, 500)



predicted_vec = nn.predict(x_test)
predicted = np.argmax(predicted_vec, axis=1)

elapsed = time.time() - start
print("Elapsed time:", elapsed)
print(f"Stopped at epoch {stopped}")

def print_results(predicted, y_test):
	boolean_predicted = predicted.copy()
	boolean_test = y_test.copy()
	boolean_predicted[boolean_predicted > 0] = 1
	boolean_test[boolean_test > 0] = 1
	boolean_accuracy = np.mean(boolean_predicted == boolean_test)

	print("Boolean clf accuracy: {}".format(boolean_accuracy))
	print("Boolean f-2 scores: {}".format(fbeta_score(boolean_test, boolean_predicted, average=None, beta=2)))

	print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))

	f2_scores = fbeta_score(y_test, predicted, average=None, beta=2)

	print("F-2 scores: {}  | Average: {}".format(f2_scores, np.mean(f2_scores)))

	print("Confusion matrix: \n{}".format(confusion_matrix(y_test, predicted)))

print_results(predicted, y_test)

## Organise documents by standard deviation
#Divide each number in predicted_vec by the sum 
sums = np.sum(predicted_vec, 1)
sums = np.linalg.norm(predicted_vec, 1)
divided = (predicted_vec.T/sums).T
stds = np.std(divided, 1)
indices = np.argsort(stds)

def smooth(keep, arr):
	smoothed = []
	previous = arr[0]
	for i in range(len(arr)):
		previous = keep*previous + (1-keep)*arr[i]
		smoothed.append(previous)
	return smoothed

def to_1_interval(arr):
	minimum = min(arr)
	maximum = max(arr)
	new_arr = []
	for value in arr:
		new_arr.append((value-minimum)/(maximum-minimum))
	return new_arr

	
import matplotlib.pyplot as plt

smoothed = smooth(0.96, stds[indices])
for i in range(len(smoothed)-1):
	delta = smoothed[i+1] - smoothed[i]
	if delta < 0.0005 and i > 20:
		n = i
		print(f"delta found! {delta} -- n: {n}")
		break

def show_confidence_graph():
	accuracies = []
	for i in range(len(stds[indices])):
		p = np.argmax(predicted_vec[indices][i:], 1)
		y = y_test[indices][i:]
		accuracies.append(np.mean(p == y))
	#plt.plot(to_1_interval(smooth(0.96, accuracies)), c='blue')
	plt.plot(to_1_interval(accuracies), c='blue')
	plt.plot(to_1_interval(smooth(0.85, stds[indices])), c='orange')
	#plt.scatter([n], [to_1_interval(smooth(0.90, accuracies))[n]])
	plt.show()

#show_confidence_graph()

n = 55
predicted_vec = predicted_vec[indices][n:]
y_test_high_confidence = y_test[indices][n:]

predicted_high_confidence = np.argmax(predicted_vec, 1)
print("High confidence results:")
print_results(predicted_high_confidence,  y_test_high_confidence)


####################################
import matplotlib.pyplot as plt
	
def show_overfit_plot():
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.legend(['train','test'], loc='upper left')
	plt.show()

show_overfit_plot()

def show_variance_plot():
	explained = preprocessing.named_steps['pca'].explained_variance_
	cumulative = [np.sum(explained[:i]) for i in range(len(explained))]
	#plt.plot(explained)
	plt.plot(cumulative)
	plt.legend(['explained variance','cumulative explained variance'], loc='upper left')
	plt.show()














