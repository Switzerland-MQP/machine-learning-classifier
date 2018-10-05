###############################################
## Final classifier pipeline                 ##
## Authors: Griffin Bishop, Sam Pridotkas,   ##
## Harry Sadoyan, Leo Grande                 ##
###############################################

# Make it so that if the document predictor predicts non-personal, we don't do category or line testing
#Make it so that if the document predictor predicts personal, then don't predict sensitive categories.


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
import pickle
### End imports ###





### Begin models ###
def load_keras_model(json_path, weights_path):
	json_file = open(json_path, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(weights_path)
	return loaded_model

def confidence(predicted_vec):
	predicted_vec = np.array(predicted_vec)
	norm = np.linalg.norm(np.sum(predicted_vec))
	divided = (predicted_vec.T/norm).T
	stdev = np.std(divided)
	return stdev


print("Loading models from disk...")
preprocessing = pickle.load(open("./models/preprocessing.pickle", "rb"))
document_clf = load_keras_model("./models/document-clf/model.json",
																"./models/document-clf/model.h5")
category_clf = load_keras_model("./models/category-clf/model.json",
																"./models/category-clf/model.h5")
#TODO: load line clf
print("Done loading models from disk.")


loaded_documents = load_files('./documents/text_documents', shuffle=False)

# TODO: line preprocessing and prediction
for i in range(len(loaded_documents.data)):
	path = loaded_documents.filenames[i]
	text = loaded_documents.data[i]
	text_pca = preprocessing.transform([text])

	#Predict document class
	predicted_vec = document_clf.predict(text_pca)
	predicted_class = np.argmax(predicted_vec, axis=1)

	#Predict individual categories
	predicted_categories = category_clf.predict([text_pca])[0]
	high_probability_categories = []
	for i in range(22):
		category = utils.all_categories_dict[i+1]
		cutoff = 0.002 # TODO: make this cutoff programmatic
										# We're going to have to just pick a number for each category
		if predicted_categories[i+1] > 0.002:
			high_probability_categories.append(category)

		# Could also just take the highest 5 categories

	# TODO: Line prediction

	print(f"File path: {path}")
	print(f"Predicted class: {predicted_class} with confidence {confidence(predicted_vec)}")
	print(f"High probability categories: {high_probability_categories}")

	# TODO: put original file in new directory

