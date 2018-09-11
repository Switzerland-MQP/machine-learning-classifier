import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

numpy.random.seed(7)

top_words = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)

#Truncate and pad sequences
max_length = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

embedding_vector_length= 32

def create_model(top_words, embedding_vector_length, max_length):
	model = Sequential()
	model.add(Embedding(top_words, embedding_vector_length, input_length=max_length))
	model.add(LSTM(100, dropout=0, recurrent_dropout=0))
	model.add(Dense(1, activation='sigmoid')) #This will need to be 3, softmax
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Will need to be categorical_crossentropy
	return model

model = create_model(top_words, embedding_vector_length, max_length)
print(model.summary())
model.fit(x_train, y_train, epochs=3, batch_size=64)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
	
