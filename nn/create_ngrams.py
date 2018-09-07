import time
start = time.time()

from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

import numpy as np


documents = load_files('../TEXTDATA/', shuffle=False)
x_train, x_test, y_train, y_test = train_test_split(
    documents.data, documents.target, test_size=0.3
)

preprocessing = Pipeline([('count', CountVectorizer(ngram_range=(1,3))),
												  ('tfidf', TfidfTransformer()),
													('pca', TruncatedSVD(n_components=430))])
preprocessing.fit(documents.data)
data = preprocessing.transform(documents.data)

np.save('x.npy', data)
np.save('y.npy', documents.target)
print("Saved preprocessed files!")

