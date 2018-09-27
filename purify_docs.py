'''
Run this on text files to replace common misspellings & remove stopwords found within our documents
'''
import collections
import fileinput

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os
import shutil
from shutil import copyfile
import sys
import re

''' Word Counts for each document '''

total_list = []

def word_count(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = root + '/' + file
            words = re.findall('\w+', open(full_path).read().lower())
            count = (collections.Counter(words))
            count_view = [(v,k) for k,v in count.items()]
            count_view.sort(reverse=True)
            for v,k in count_view:
                print(f"{k}: {v}")
            print('=======================================================')

    
'''
Common misspellings
'''

def if_misspell_then_replace(word, misspell_array, full_path, filedata, replace_word):
    if word in misspell_array:
        newdata = re.sub(r'\b'+word+r'\b'.format(word), replace_word, filedata)
        f = open(full_path, 'w')
        f.write(newdata)
        f.close()
        filedata = newdata
    return filedata


def clean_misspellings_and_standardize(directory):
    # German stop words
    stop_words = set(stopwords.words('german'))
    
    # Original: herr
    herr_misspell = ['hurr', 'her', 'herrr', 'hur', 'l-lerr', 'lerr', 'herrn', 'hern']
    
    # Original: staatsangehörigkeit
    nationality_missphone = ['nationalitat', 'nationalität', 'staatszugehörigkeit', 'staatsangehsrigkeit', 'staatsangehdrigkeit', 'staatsangehérigkeit']
    
    # Original: geb and geb.
    dob_misspell = ['gab.']
    
    # Original: Telefon
    phone_misspell = ['teleion', 'televon', 'telefonie', 'telefonieren', 'telefono', 'telefone', 'telefun', 'telefonnummer', 'telfon', 'telfone;', 'tlefone', 'telefo']
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = root + '/' + file
            
            f = open(full_path,'r')
            filedata = f.read()
            f.close()
            
            for word in filedata.split():
                filedata = if_misspell_then_replace(word, herr_misspell, full_path, filedata, 'herr')
                filedata = if_misspell_then_replace(word, nationality_missphone, full_path, filedata, 'staatsangehörigkeit')
                filedata = if_misspell_then_replace(word, dob_misspell, full_path, filedata, 'geb.')
                filedata = if_misspell_then_replace(word, phone_misspell, full_path, filedata, 'telefon')
                filedata = if_misspell_then_replace(word, stop_words, full_path, filedata, '')


clean_misspellings_and_standardize('./PURIFIED_TEXT_DATA')
