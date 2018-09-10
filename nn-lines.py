import numpy as np
import time
start = time.time()


#  from scipy.stats import randint as sp_randint
from sklearn.decomposition import TruncatedSVD
from scipy.stats import randint as sp_randint

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, fbeta_score

from sklearn.pipeline import Pipeline
import utils

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.regularizers import l1
import keras.backend as K

import matplotlib.pyplot as plt
	
def show_overfit_plot():
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.legend(['train','test'], loc='upper left')
	plt.show()



documents = utils.load_dirs_custom([
    './TAGGED_DATA/SENSITIVE_DATA/html-tagged',
    './TAGGED_DATA/PERSONAL_DATA/html-tagged',
    './TAGGED_DATA/NON_PERSONAL_DATA'
])

doc_train, doc_test, = utils.document_test_train_split(
    documents, 0.20
)

print("Doc train: ", len(doc_train))
print("Doc test: ", len(doc_test))
x_train, y_train = utils.convert_docs_to_lines(doc_train)
x_test, y_test = utils.convert_docs_to_lines(doc_test)

print("Convert lines timing {}".format(time.time() - start))

preprocessor = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('pca', TruncatedSVD(n_components=430))])
preprocessor.fit(x_train)
x_train, x_test = (preprocessor.transform(x_train), preprocessor.transform(x_test))
print("Finished data preprocessing - {} elapsed".format(time.time()-start))

def mse(y_true, y_pred):
	x = K.square(y_pred-y_true)
	return K.dot(K.variable(K.cast_to_floatx(np.array([0.4, 1, 1.5]))), x)
	


def create_model():
	input_shape = x_train.shape[1]

	nn = Sequential()
	nn.add(Dense(16, activation='relu', input_shape=(input_shape,)))
	nn.add(Dropout(0.25))
	#nn.add(Dense(8, activation='relu'))
	#nn.add(Dropout(0.5))
	nn.add(Dense(3,  activation='softmax', name="out_layer"))
	#nn.compile(loss= 'categorical_crossentropy',
	nn.compile(loss=mse, optimizer='adam')
	return nn

print("Begin fitting network")
nn = create_model()

y_train = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)

def fit(batch_size, epochs):
	global x_train, y_train, x_test, y_test_onehot
	return nn.fit(x_train, y_train,
								batch_size=batch_size,
								epochs=epochs,
								verbose=2,
								validation_data=(x_test, y_test_onehot))

history = fit(1000, 35)



elapsed = time.time() - start
print("Elapsed time:", elapsed)

show_overfit_plot()



documents_predicted = []
documents_target = []
all_predicted_lines = []
all_target_lines = []
for doc in doc_test:
    predicted_lines = nn.predict(preprocessor.transform(doc.data))
    predicted_lines = np.argmax(predicted_lines, axis=1)

    all_predicted_lines += list(predicted_lines)
    all_target_lines += list(doc.targets)

    predicted_doc = utils.classify_doc(predicted_lines)
    documents_predicted.append(predicted_doc)
    documents_target.append(doc.category)


print("Line by Line ")
print("Confusion Matrix: \n{}".format(
    confusion_matrix(all_target_lines, all_predicted_lines)
))

accuracy = fbeta_score(
    all_target_lines,
    all_predicted_lines,
    average=None,
    beta=2
)
print("Accuracy: {}".format(accuracy))


doc_accuracy = fbeta_score(
    documents_target,
    documents_predicted,
    average=None,
    beta=2
)

print("Document Accuracy: {}".format(doc_accuracy))

print("Document Confusion Matrix: \n{}".format(
    confusion_matrix(documents_target, documents_predicted)
))

