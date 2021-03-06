import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

np.random.seed(7)

import matplotlib.pyplot as plt
def show_overfit_plot(history):
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.legend(['train','test'], loc='upper left')
	plt.show()





documents = load_files('../TEXTDATA/', shuffle=True)

vocabulary_size = 100000 # Should be 150,000
encoded_docs = [one_hot(str(d), vocabulary_size) for d in documents.data]

#Truncate and pad sequences
max_length = 2000
padded_docs = sequence.pad_sequences(encoded_docs, maxlen=max_length)

x_train, x_test, y_train, y_test = train_test_split(
	padded_docs, documents.target, test_size=0.3
)

y_train = np.where((y_train == 2) | (y_train == 1), 1, 0)
y_test = np.where((y_test == 2) | (y_test == 1), 1, 0)


embedding_vector_length = 64

def create_model(top_words, embedding_vector_length, max_length):
	model = Sequential()
	model.add(Embedding(top_words, embedding_vector_length, input_length=max_length))
	model.add(Flatten())
	model.add(Dense(490, activation="relu"))
	model.add(Dropout(0.49))
	model.add(Dense(115, activation="relu"))
	model.add(Dropout(0.005))
	model.add(Dense(370, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(118, activation="relu"))
	model.add(Dropout(0.185))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	# Will need to be categorical_crossentropy
	return model

model = create_model(vocabulary_size, embedding_vector_length, max_length)
print(model.summary())
history = model.fit(x_train, y_train, epochs=10, batch_size=32,
								validation_data=(x_test, y_test))


scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


y_predicted = model.predict(x_test)

score = fbeta_score(y_predicted, y_test, average=None, beta=2)
print(f"F2-scores: {score}")
show_overfit_plot(history)




