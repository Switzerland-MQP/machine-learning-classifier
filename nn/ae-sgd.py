from sklearn.datasets import load_files

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier


from keras.models import Sequential, Model
from keras.layers import Dense
from keras.regularizers import l1

from keras import backend as K

import numpy as np

documents = load_files('../TEXTDATA/', shuffle=False)

# Split remainder into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    documents.data, documents.target, test_size=0.15
)

#import code
#code.interact(local=locals())

count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

# Test data transformations
X_test_count = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_count)

# AutoEncoder
input_shape = X_train_tfidf.shape[1]


ae = Sequential()
ae.add(Dense(1024, activation='relu', input_shape=(input_shape,)))
ae.add(Dense(64, activation='relu', name="bottleneck", input_shape=(input_shape,),
					activity_regularizer=l1(10e-6)))
#ae.add(Dense(512,  activation='relu'))
ae.add(Dense(1024, activation='relu', input_shape=(input_shape,)))
ae.add(Dense(input_shape,  activation='relu', name="out_layer"))
ae.compile(loss= 'mean_squared_error',
           optimizer='adam',
           metrics=['mean_squared_logarithmic_error'])

print("Begin fitting network")

ae.fit(X_train_tfidf, X_train_tfidf,
                 batch_size=196,
                 epochs=2,
                 validation_data=(X_test_tfidf, X_test_tfidf))

print("======================\n Evaluating autoencoder")
metrics = ae.evaluate(X_test_tfidf, X_test_tfidf,verbose=1)

print(metrics)



encoder = Model(ae.input, ae.get_layer('bottleneck').output)


#np.savez('autoencoded_data.npz', x=encoder.predict(X_train_tfidf), y=y_train)

print("Done encoding vectors")

encoded_train = encoder.predict(X_train_tfidf)
encoded_test = encoder.predict(X_test_tfidf)

pca = TruncatedSVD(n_components=64)
pca.fit(X_train_tfidf)
pca_train = pca.transform(X_train_tfidf)
pca_test = pca.transform(X_test_tfidf)



text_clf1 = Pipeline([('clf', SGDClassifier(loss='hinge', penalty='none', learning_rate='optimal', alpha=1e-4, epsilon=0.1, max_iter=1000, tol=None, shuffle=True)),
])

text_clf1.fit(pca_train, y_train)
predicted = text_clf1.predict(pca_test)

boolean_predicted = predicted.copy()
boolean_test = y_test.copy()
boolean_predicted[boolean_predicted > 0] = 1
boolean_test[boolean_test > 0] = 1
boolean_accuracy = np.mean(boolean_predicted == boolean_test)

print("Boolean clf accuracy: {}".format(boolean_accuracy))
print("Boolean f-2 scores: {}".format(fbeta_score(boolean_test, boolean_predicted, average=None, beta=2)))

print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))

print("F-2 scores: {}".format(fbeta_score(y_test, predicted, average=None, beta=2)))

print("Confusion matrix: \n{}".format(confusion_matrix(y_test, predicted)))






text_clf2 = Pipeline([('clf', SGDClassifier(loss='hinge', penalty='none', learning_rate='optimal', alpha=1e-4, epsilon=0.1, max_iter=1000, tol=None, shuffle=True)),
])

text_clf2.fit(encoded_train, y_train)
predicted = text_clf2.predict(encoded_test)

boolean_predicted = predicted.copy()
boolean_test = y_test.copy()
boolean_predicted[boolean_predicted > 0] = 1
boolean_test[boolean_test > 0] = 1
boolean_accuracy = np.mean(boolean_predicted == boolean_test)

print("Boolean clf accuracy: {}".format(boolean_accuracy))
print("Boolean f-2 scores: {}".format(fbeta_score(boolean_test, boolean_predicted, average=None, beta=2)))

print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))

print("F-2 scores: {}".format(fbeta_score(y_test, predicted, average=None, beta=2)))

print("Confusion matrix: \n{}".format(confusion_matrix(y_test, predicted)))





