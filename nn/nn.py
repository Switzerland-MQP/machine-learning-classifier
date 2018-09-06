import time
start = time.time()

from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import fbeta_score, confusion_matrix 


from keras.models import Sequential, Model
from keras.layers import Dense
from keras.regularizers import l1

from keras import backend as K

import numpy as np

from keras.utils import np_utils


documents = load_files('../TEXTDATA/', shuffle=False)

# Split remainder into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    documents.data, documents.target, test_size=0.20
)


count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

# Test data transformations
X_test_count = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_count)

pca = TruncatedSVD(n_components=800)
X_train_pca = pca.fit_transform(X_train_tfidf)
X_test_pca = pca.transform(X_test_tfidf)


input_shape = X_train_pca.shape[1]


ae = Sequential()
ae.add(Dense(8192, activation='relu', input_shape=(input_shape,),
					activity_regularizer=l1(10e-6)))
#ae.add(Dense(64, activation='relu', name="bottleneck", input_shape=(input_shape,),
#					activity_regularizer=l1(10e-6)))
#ae.add(Dense(512,  activation='relu'))
#ae.add(Dense(1024, activation='relu', input_shape=(input_shape,)))
ae.add(Dense(3,  activation='softmax', name="out_layer"))
ae.compile(loss= 'categorical_crossentropy',
           optimizer='adam',
           metrics=['mean_squared_logarithmic_error'])

print("Begin fitting network")

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_test_onehot = y_test.copy()

ae.fit(X_train_pca, y_train,
                 batch_size=196,
                 epochs=50,
								 verbose=2,
                 validation_data=(X_test_pca, y_test))

predicted = ae.predict(X_test_pca)
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













