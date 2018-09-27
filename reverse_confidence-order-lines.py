# This file takes in a directory name and organizes the unlabeled files by 
# our confidence in our average line prediction for that file.

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
import utils_normal as utils

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.regularizers import l1
import keras.backend as K

import matplotlib.pyplot as plt

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

documents = utils.n_gram_documents_range(documents, 8, 8)

doc_train, doc_test, = utils.document_test_train_split(
    documents, 0.20
)

print("Doc train: ", len(doc_train))
print("Doc test: ", len(doc_test))
x_train, y_train = utils.convert_docs_to_lines(doc_train)
x_test, y_test = utils.convert_docs_to_lines(doc_test)

y_train = np.where((y_train == 2) | (y_train == 1), 1, 0)
y_test = np.where((y_test == 2) | (y_test == 1), 1, 0)

print("Convert lines timing {}".format(time.time() - start))

preprocessor = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('pca', TruncatedSVD(n_components=430))])
preprocessor.fit(x_train)
x_train, x_test = (preprocessor.transform(x_train), preprocessor.transform(x_test))
print("Finished data preprocessing - {} elapsed".format(time.time()-start))
	
def create_model():
	input_shape = x_train.shape[1]

	nn = Sequential()
	nn.add(Dense(16, activation='relu', input_shape=(input_shape,)))
	nn.add(Dropout(0.25))
	#nn.add(Dense(8, activation='relu'))
	#nn.add(Dropout(0.5))
	nn.add(Dense(1,  activation='sigmoid', name="out_layer"))
	#nn.compile(loss= 'categorical_crossentropy',
	nn.compile(loss='binary_crossentropy', optimizer='adam')
	return nn

print("Begin fitting network")
nn = create_model()

#y_train = np_utils.to_categorical(y_train)
#y_test_onehot = np_utils.to_categorical(y_test)

def fit(batch_size, epochs):
	global x_train, y_train, x_test, y_test
	return nn.fit(x_train, y_train,
								batch_size=batch_size,
								epochs=epochs,
								verbose=2,
								validation_data=(x_test, y_test))

history = fit(1000, 30)



elapsed = time.time() - start
print("Elapsed time:", elapsed)

#show_overfit_plot()


# Do unlabeled documents
unlabeled_documents = utils.load_dirs_custom([
	'./UNLABELED_DATA/PERSONAL'
])
print("Unlabeled docs:", len(unlabeled_documents))
unlabeled_documents = utils.n_gram_documents_range(unlabeled_documents, 8, 8)
 





documents_predicted = []
documents_target = []
all_predicted_lines = []
all_target_lines = []
document_confidences = []
for doc in unlabeled_documents:
    print(doc.path)
    try:
        feature_vectors = preprocessor.transform(doc.data)
    except:
        print(f"Found file with low length, skipping: {doc.data}")
        continue
    predicted_lines = nn.predict(feature_vectors)
		

    predicted_lines_confs = np.array([x for x in map(lambda x: x[0], list(predicted_lines))])
    document_confidence = np.mean(predicted_lines_confs)
    document_confidences.append(document_confidence)


    all_predicted_lines += list(predicted_lines)
    doc.targets = np.where((doc.targets == 2) | (doc.targets == 1), 1, 0)
    all_target_lines += list(doc.targets)


    predicted_doc = utils.classify_doc(predicted_lines)
    documents_predicted.append(predicted_doc)
    documents_target.append(doc.category)

document_confidences = np.array(document_confidences)
indices = np.argsort(document_confidences)

lengths = []

for i in range(len(indices)):
	doc_path = unlabeled_documents[indices[i]].path
	confidence = document_confidences[indices[i]]
	filename = f"{i}-{confidence:.3f}-{doc_path[32:]}"
	print(filename)

	f = open('./confidence_out/personal_line_level/'+filename, "w+")
	infile = open(doc_path)
	write_data = infile.read()
	f.write(write_data)
	lengths.append(len(write_data))
	f.close()

plt.plot(document_confidences[indices])
#plt.plot(lengths)
plt.show()
