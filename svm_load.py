import numpy as np
from scipy.stats import randint as sp_randint
import utils
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split


def load_predict(directory):
    clf = joblib.load('svm_trained.joblib')
    documents = utils.load_dirs_custom(directory)
    documents = utils.n_gram_documents_range(documents, 5, 6)
    documents = np.array(documents)
    doc_test = utils.convert_docs_to_lines(documents)
    predicted_lines = []
    target_lines = []
    for doc in doc_test:
        lines = clf.predict(doc.data)
        predicted_lines += list(lines)
    return predicted_lines
    
     


