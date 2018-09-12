import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os
import shutil
from shutil import copyfile
import sys
import re


stop_words = set(stopwords.words('german'))

dirs = ["./SENSITIVE_DATA/html-tagged/", "./PERSONAL_DATA/html-tagged/", "./NON_PERSONAL_DATA/"]
cleaned_words = []
for direct in dirs:
    documents = os.listdir(direct)
    for doc in documents:
        path = direct + doc
        file = open(path, 'r')
        text = file.read()
        words = word_tokenize(text)
        for w in words:
            if w not in stop_words:
                cleaned_words.append(w)
        file.write(clean_words)
        file.close()

print(cleaned_words)



