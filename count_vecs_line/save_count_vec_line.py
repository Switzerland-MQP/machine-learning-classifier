import numpy as np
#  from scipy.stats import randint as sp_randint
from sklearn.decomposition import TruncatedSVD
from scipy.stats import randint as sp_randint

# Models to try
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#  from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score

from sklearn.pipeline import Pipeline
import utils

documents = utils.load_dirs_custom([
    '../SENSITIVE_DATA/html-tagged',
    '../PERSONAL_DATA/html-tagged',
    '../NON_PERSONAL_DATA'
])

X_info, y_info = utils.convert_docs_to_lines(documents)

count_vect = CountVectorizer()
doc_count = count_vect.fit_transform(X_info)
print(doc_count)
print('==============')
print(y_info)
np.save("count_x_line", doc_count)
np.save("count_y_line", y_info)

