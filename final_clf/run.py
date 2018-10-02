###############################################
## Final classifier pipeline                 ##
## Authors: Griffin Bishop, Sam Pridotkas,   ##
## Harry Sadoyan, Leo Grande                 ##
###############################################

# Pipeline steps:
#
# 1. Convert directory of documents (pdfs, excel, word, text) into text files
# 
#
#
#


### Begin imports ###

import time
start = time.time()
from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from sklearn.metrics import fbeta_score, confusion_matrix 

import numpy as np
from keras.utils import np_utils
from keras.models import model_from_json
import utils
### End imports ###




### Begin models ###
CATEGORY_F2_SCORE_CUTOFF = 0.5


def load_keras_model(json_path, weights_path):
	json_file = open(json_path, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(weights_path)
	return loaded_model


print("Beggining document preprocessing for document level classification...")
documents = load_files('./documents/text_documents/', shuffle=False)
x = documents.data
preprocessing = Pipeline([('count', CountVectorizer()),
												  ('tfidf', TfidfTransformer()),
													('pca', TruncatedSVD(n_components=430))])
preprocessing.fit(x)
x = preprocessing.transform(x)
print("Finished document preprocessing - document level classifcation - {} elapsed".format(time.time()-start))



##########
print("Loading document classifier model from disk")
document_clf = load_keras_model("./pipeline/models/document-clf/model.json",
																"./pipeline/models/document-clf/model.h5")
predicted_vec = document_clf.predict(x)
predicted = np.argmax(predicted_vec, axis=1)

print("Done predicting document classifications")


##########
print("Loading individual category classifer model from disk")
category_clf = load_keras_model("./pipeline/models/category-clf/model.json",
																"./pipeline/models/category-clf/model.h5")
predicted = category_clf.predict(x)

for i in range(22):
	category = utils.all_categories_dict[i+1]
	predicted_category = predicted[:,i+1]
	y_category = y_test[:,i+1]

	cutoff = 0.002 # TODO: make this cutoff programmatic 

	predicted_category = np.where(predicted_category > cutoff, 1, 0)
	
# TODO: assign highest probability categories to each document



#########
print("Loading line classifier model from disk")




