'''
Run this on text files to replace common misspellings found within our documents
'''
import os
import re
import collections
import fileinput


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
        newdata = filedata.replace(word, replace_word)
        f = open(full_path, 'w')
        f.write(newdata)
        f.close()
        print(word)
        print(replace_word)


def clean_misspellings_and_standardize(directory):
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
                if_misspell_then_replace(word, herr_misspell, full_path, filedata, 'herr')
                if_misspell_then_replace(word, nationality_missphone, full_path, filedata, 'staatsangehörigkeit')
                if_misspell_then_replace(word, dob_misspell, full_path, filedata, 'geb.')
                if_misspell_then_replace(word, phone_misspell, full_path, filedata, 'telefon')

