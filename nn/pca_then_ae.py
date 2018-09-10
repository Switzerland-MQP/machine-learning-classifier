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

from keras import backend as K

import numpy as np

from keras.utils import np_utils
import matplotlib.pyplot as plt
	
def show_overfit_plot(history):
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.legend(['train','test'], loc='upper left')
	plt.show()

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



"""
documents = load_files('../TEXTDATA/', shuffle=False)
x_train, x_test, y_train, y_test = train_test_split(
    documents.data, documents.target, test_size=0.3
)

preprocessing = Pipeline([('count', CountVectorizer()),
												  ('tfidf', TfidfTransformer()),
													('pca', TruncatedSVD(n_components=1000))])
preprocessing.fit(x_train)
x_train, x_test = (preprocessing.transform(x_train), preprocessing.transform(x_test))
"""
x_train = np.load('./npy/x_train.npy')
x_test =  np.load('./npy/x_test.npy')
y_train = np.load('./npy/y_train.npy')
y_test =  np.load('./npy/y_test.npy')



def create_autoencoder(x_train):
	input_shape = x_train.shape[1] #430 PCs
	ae = Sequential()
	ae.add(Dense(64, activation='relu', input_shape=(input_shape,), name="bottleneck"))
	ae.add(Dense(input_shape,  activation='relu', name="out_layer"))
	ae.compile(loss= 'mean_squared_error',
           optimizer='adam',
           metrics=['mean_squared_logarithmic_error'])

	encoder = Model(ae.input, ae.get_layer('bottleneck').output)
	
	return (ae, encoder)

ae, encoder = create_autoencoder(x_train)

ae_history = ae.fit(x_train, x_train,
                 batch_size=196,
                 epochs=200,
								 verbose=0,
                 validation_data=(x_test, x_test))
x_train, x_test = (encoder.predict(x_train), encoder.predict(x_test))

input_shape = x_train.shape[1]


nn = Sequential()
nn.add(Dense(256, activation='relu', input_shape=(input_shape,)))
nn.add(Dropout(0.2))
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
								validation_data=(x_test, y_test_onehot))

history = fit(196, 600)
print("Val Loss:{}".format(history.history['val_loss'][-1]))


predicted = nn.predict(x_test)
predicted = np.argmax(predicted, axis=1)

elapsed = time.time() - start
print("Elapsed time:", elapsed)

print_results(predicted, y_test)


####################################
show_overfit_plot(history)

def show_variance_plot():
	explained = preprocessing.named_steps['pca'].explained_variance_
	cumulative = [np.sum(explained[:i]) for i in range(len(explained))]
	#plt.plot(explained)
	plt.plot(cumulative)
	plt.legend(['explained variance','cumulative explained variance'], loc='upper left')
	plt.show()

