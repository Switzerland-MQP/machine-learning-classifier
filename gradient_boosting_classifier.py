import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_files
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

text_data = load_files('TEXTDATA/', shuffle=True)

# Split remainder into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    text_data.data, text_data.target, test_size=0.20
)

count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

pca = TruncatedSVD(n_components=15)
pca.fit(X_train_tfidf)

X_train_pca = pca.transform(X_train_tfidf)


# Test data transformations
X_test_count = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_count)
X_test_pca = pca.transform(X_test_tfidf)

# Try Gradient Boosting Below
g_parameters = {
    "n_estimators": 70,
    "max_depth": 12,
    "subsample": 0.8,
    "max_features": 6,
    "learning_rate": 0.3,
    "random_state": 3,
    "min_samples_leaf": 3,
    "min_samples_split": 200
}

gclf = GradientBoostingClassifier(**g_parameters)
gclf.fit(X_train_pca, y_train)
acc = gclf.score(X_test_pca, y_test)

print("Parameters used: " + str(g_parameters))
print("Accuracy: {:.6f}".format(acc))

