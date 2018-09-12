import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.manifold import TSNE


import utils


documents = utils.load_dirs_custom([
	'./TAGGED_DATA/SENSITIVE_DATA/html-tagged',
	'./TAGGED_DATA/PERSONAL_DATA/html-tagged',
	'./TAGGED_DATA/NON_PERSONAL_DATA'
])

documents = utils.n_gram_documents_range(documents, 7, 8)

doc_train, doc_test = utils.document_test_train_split(
	documents, 0.5
)

x_train, y_train = utils.convert_docs_to_lines(doc_train)
x_test, y_test = utils.convert_docs_to_lines(doc_test)

order = np.arange(len(x_train))
np.random.shuffle(order)

n = 2000

x_train, y_train = (x_train[order][:n], y_train[order][:n])



"""
documents = load_files('./TEXTDATA/')
x_train, x_test, y_train, y_test = train_test_split(
	documents.data, documents.target, test_size=0
)
"""

reducer = Pipeline([('count', CountVectorizer(ngram_range=(1,2))),
										('tfidf', TfidfTransformer()),
										('pca', TruncatedSVD(n_components=50)),
										('tsne', TSNE(n_components=2, verbose=1, perplexity=40, n_iter=6000))])
x_train_reduced = reducer.fit_transform(x_train)

plt.scatter(x_train_reduced[:,0], x_train_reduced[:,1], c=y_train)
plt.show()
