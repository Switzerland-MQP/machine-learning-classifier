import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_files
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

import utils

documents = utils.load_dirs_custom([
	'./TAGGED_DATA/SENSITIVE_DATA/html-tagged',
	'./TAGGED_DATA/PERSONAL_DATA/html-tagged',
	'./TAGGED_DATA/NON_PERSONAL_DATA'
])

doc_train, doc_test = utils.document_test_train_split(
	documents, 0.2
)

x_train, y_train = utils.convert_docs_to_lines(doc_train)
x_test, y_test = utils.convert_docs_to_lines(doc_test)

reducer = Pipeline([('count', CountVectorizer()),
										('tfidf', TfidfTransformer()),
										('pca', TruncatedSVD(n_components=2))])

x_train_reduced = reducer.fit_transform(x_train)
x_test_reduced = reducer.transform(x_test)

plt.scatter(x_train_reduced[:,0], x_train_reduced[:,1], c=y_train)
plt.show()
