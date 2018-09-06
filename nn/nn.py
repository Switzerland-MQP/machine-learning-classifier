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


documents = load_files('../TEXTDATA/', shuffle=False)
x_train, x_test, y_train, y_test = train_test_split(
    documents.data, documents.target, test_size=0.3
)

preprocessing = Pipeline([('count', CountVectorizer()),
												  ('tfidf', TfidfTransformer()),
													('pca', TruncatedSVD(n_components=400))])
preprocessing.fit(x_train)
x_train, x_test = (preprocessing.transform(x_train), preprocessing.transform(x_test))



print("Finished data preprocessing - {} elapsed".format(time.time()-start))


input_shape = x_train.shape[1]


nn = Sequential()
nn.add(Dense(2048, activation='relu', input_shape=(input_shape,))),
nn.add(Dropout(0.7))
nn.add(Dense(1024, activation='relu'))
nn.add(Dropout(0.7))
nn.add(Dense(3,  activation='softmax', name="out_layer"))
nn.compile(loss= 'categorical_crossentropy',
           optimizer='adam',
           metrics=['mean_squared_logarithmic_error'])

print("Begin fitting network")

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_test_onehot = y_test.copy()

history = nn.fit(x_train, y_train,
                 batch_size=196,
                 epochs=40,
								 verbose=2,
                 validation_data=(x_test, y_test))

predicted = nn.predict(x_test)
predicted = np.argmax(predicted, axis=1)
y_test = np.argmax(y_test, axis=1)

elapsed = time.time() - start
print("Elapsed time:", elapsed)


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

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','test'], loc='upper left')
plt.show()










