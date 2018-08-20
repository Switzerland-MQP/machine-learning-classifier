import numpy as np

from scipy.stats import randint as sp_randint
from sklearn.datasets import load_files
from sklearn.decomposition import PCA

# Models to try
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


from keras.models import Sequential, Model
from keras.layers import Dense


# Import Data
moviedir = 'data/movie_reviews/txt_sentoken/'
reviews = load_files(moviedir, shuffle=True)

# Hide some data in the vault to avoid snooping
x_work, x_vault, y_work, y_vault = train_test_split(
    reviews.data, reviews.target, test_size=0.10
)

# Split remainder into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    x_work, y_work, test_size=0.10
)

count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

pca = PCA(n_components=120)
X_train_array = X_train_tfidf.toarray()
pca.fit(X_train_array)

X_train_pca = pca.transform(X_train_array)


# Plot the Principal Components
#  plt.figure()
#  plt.plot(pca.explained_variance_)
#  plt.xlabel('number of components')
#  plt.ylabel('cumulative explained variance')
#  plt.show()

# Test data transformations
X_test_count = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_count)
X_test_pca = pca.transform(X_test_tfidf.toarray())

clf = RandomForestClassifier()

# specify parameters and distributions to sample from
parameters_rand = {
    "n_estimators": sp_randint(300, 2000),
    "max_depth": [3, None],
    "max_features": sp_randint(1, 11),
    "min_samples_split": sp_randint(2, 11),
    "min_samples_leaf": sp_randint(1, 11),
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"]
}

# run randomized search
# Accuracy should be comparable to grid search, but runs much much faster
n_iter_search = 40
random_search = RandomizedSearchCV(clf, param_distributions=parameters_rand,
                                   n_iter=n_iter_search,
                                   n_jobs=-1)

random_search.fit(X_train_pca, y_train)
print(random_search.cv_results_)
predicted = random_search.predict(X_test_pca)


print("PCA with random forest")
print("Accuracy: {}".format(np.mean(predicted == y_test)))

# AutoEncoder

input_shape = X_train_tfidf.shape[1]

ae = Sequential()
ae.add(Dense(512,  activation='elu', input_shape=(input_shape,)))
ae.add(Dense(128,  activation='relu'))
ae.add(Dense(1,    activation='linear', name="bottleneck"))
ae.add(Dense(128,  activation='relu'))
ae.add(Dense(512,  activation='elu'))
ae.add(Dense(input_shape,  activation='sigmoid', name="out_layer"))
ae.compile(loss='binary_crossentropy',
           optimizer='adadelta',
           metrics=['accuracy'])


history = ae.fit(X_train_tfidf, X_train_tfidf,
                 batch_size=128,
                 epochs=5,
                 verbose=1,
                 validation_data=(X_test_tfidf, X_test_tfidf))


encoder = Model(ae.input, ae.get_layer('bottleneck').output)

encoder.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

encoder.fit(X_train_tfidf, y_train,
            nb_epoch=10,
            batch_size=128,
            shuffle=True,
            validation_data=(X_test_tfidf, y_test))

print("Autoencoder")
scores = encoder.evaluate(X_test_tfidf, y_test, verbose=1)
print("Accuracy: ", scores[1])
