###############################################
## Final classifier pipeline                 ##
## Authors: Griffin Bishop, Harry Sadoyan,   ##
## Sam Pridotkas, Leo Grande                 ##
###############################################

# Pipeline steps:
#
# 1. Convert directory of documents (pdfs, excel, word, text) into text files
#
#
#
#

# 1. Make it so that if the document predictor predicts non-personal, we don't do category or line testing
# 1a. Make it so that if the document predictor predicts personal, then don't predict sensitive categories.


########################################### Begin imports
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
import time
start = time.time()
########################################### End imports




########################################### Begin models
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


###########################################
def run_model(filepath):
    print("Loading models from disk...")
    document_preprocessing = pickle.load(open("./models/document-clf/preprocessing.pickle", "rb"))
    document_clf = load_keras_model("./models/document-clf/model.json",
                                    "./models/document-clf/model.h5")

    category_preprocessing = pickle.load(open("./models/category-clf/preprocessing.pickle", "rb"))
    cutoffs = pickle.load(open("./models/category-clf/cutoffs.pickle", "rb"))
    category_clf = load_keras_model("./models/category-clf/model.json",
                                    "./models/category-clf/model.h5")

    line_preprocessing = pickle.load(open("./models/line-clf/preprocessing.pickle", "rb"))
    line_clf = load_keras_model("./models/line-clf/model.json",
                                "./models/line-clf/model.h5")
    print("Done loading models from disk.")
    ###########################################

    #  loaded_documents = load_files('./documents/text_documents', shuffle=False)
    loaded_documents = utils.load_dir_custom(filepath)

    # Line preprocessing INCOMPLETE?
    #  loaded_line_documents = utils.load_dirs_custom(['./documents/text_documents/text_documents'])
    line_groups = utils.n_gram_documents_range(loaded_documents, 8, 8)

    results = []
    for doc in loaded_documents:
        path = doc.path
        text = doc.text
        document_pca = document_preprocessing.transform([text])
        category_pca = category_preprocessing.transform([text])

        #  Predict document class
        predicted_vec = document_clf.predict(document_pca)
        predicted_class = np.argmax(predicted_vec, axis=1)

        #Predict individual categories
        predicted_categories = category_clf.predict([category_pca])[0]
        high_probability_categories = []
        for i in range(22):
            category = utils.all_categories_dict[i+1]
            cutoff = cutoffs[category]
            if predicted_categories[i+1] > cutoff:
                high_probability_categories.append(category)

        # TODO: Line prediction

        results.append((path, predicted_class, high_probability_categories))
        print(f"File path: {path}")
        print(f"Predicted class: {predicted_class} with confidence {confidence(predicted_vec)}")
        print(f"High probability categories: {high_probability_categories}")

        # TODO: put original files in new directory corresponding to class
        # TODO: write metadata file to this new directory
    return results

