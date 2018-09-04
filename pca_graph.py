import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

import utils
sns.set()


data, target, documents = utils.load_dirs_custom([
    './SENSITIVE_DATA/html-tagged',
    './PERSONAL_DATA/html-tagged',
    './NON_PERSONAL_DATA'
])

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.20
)


count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

pca = TruncatedSVD(n_components=1000)
pca.fit(X_train_tfidf)

X_train_pca = pca.transform(X_train_tfidf)


# Test data transformations
X_test_count = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_count)
X_test_pca = pca.transform(X_test_tfidf)


plt.figure(1)
plt.subplot(121)

plt.plot(np.cumsum(pca.explained_variance_))
plt.title("PCA")
plt.ylabel("Explained Variance")
plt.xlabel("# components")

plt.subplot(122)
plt.plot(pca.explained_variance_)
plt.title("PCA Individual Explained Variance")
plt.ylabel("Explained Variance")
plt.xlabel("# components")
plt.show()

plt.tight_layout()
plt.show()

plt.plot(pca.explained_variance_)
plt.title("PCA Cumulative Explianed Variance")
plt.ylabel("Cumulative Explained Variance")
plt.xlabel("# components")
plt.show()


