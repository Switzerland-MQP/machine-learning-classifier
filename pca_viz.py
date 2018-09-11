import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

import utils

"""
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
"""

documents = load_files('./TEXTDATA/')
x_train, x_test, y_train, y_test = train_test_split(
	documents.data, documents.target, test_size=0
)


reducer = Pipeline([('count', CountVectorizer()),
										('tfidf', TfidfTransformer()),
										('pca', TruncatedSVD(n_components=2))])
x_train_reduced = reducer.fit_transform(x_train)

colors = []
for i in range(len(y_train)):
	colors.append(['green', 'red', 'purple'][y_train[i]])

plt.scatter(x_train_reduced[:,0], x_train_reduced[:,1], c=colors)
plt.show()
