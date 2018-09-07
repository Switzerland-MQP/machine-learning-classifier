from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn import metrics

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

#import code
#code.interact(local=locals())


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
                 epochs=9,
                 validation_data=(X_test_tfidf, X_test_tfidf))

print("======================\n Evaluating autoencoder")
metrics = ae.evaluate(X_test_tfidf, X_test_tfidf,verbose=1)

print(metrics)



encoder = Model(ae.input, ae.get_layer('bottleneck').output)

#x = np.vstack((encoder.predict(X_train_tfidf), encoder.predict(X_test_tfidf)))
#y = np.hstack((y_train, y_test))

#np.savez('autoencoded_data.npz', x=x, y=y)

np.savez('autoencoded_data.npz', x=encoder.predict(X_train_tfidf), y=y_train)

print("Done encoding vectors")










