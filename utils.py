import numpy as np
import os
from parser.run_parser import run_parser

personal_categories = [
    'name', 'id-number', 'location', 'online-id',
    'dob', 'phone', 'physical', 'physiological',
    'professional', 'genetic', 'mental', 'economic',
    'cultural', 'social'
]
sensitive_categories = [
    'criminal', 'origin', 'health',
    'religion', 'political', 'philosophical',
    'unions', 'sex-life', 'sex-orientation',
    'biometric'
]


def load_dirs_custom(directories):
    all_data = np.array([])
    all_target = np.array([])
    all_documents = []
    for d in directories:
        data, target, documents = load_dir_custom(d)
        all_data = np.append(all_data, data)
        all_target = np.append(all_target, target)
        all_documents += documents
    return all_data, all_target, all_documents


def load_dir_custom(directory):
    documents = read_dir(directory)

    all_data = np.array([])
    all_target = np.array([])
    for doc in documents:
        data = np.array([])
        target = np.array([])
        for line in doc.lines:
            data = np.append(data, [line.text])
            target = np.append(target, [convert_categories(line.categories)])
        doc.data = data
        doc.target = target
        doc.category = classify_doc(target)

        all_data = np.append(all_data, data)
        all_target = np.append(all_target, target)

    return all_data, all_target, documents


def classify_doc(target_array):
    if 2 in target_array:
        return 2
    elif 1 in target_array:
        return 1
    else:
        return 0


def convert_categories(categories):
    for c in sensitive_categories:
        if c in categories:
            return 2
    for c in personal_categories:
        if c in categories:
            return 1
    return 0


def read_dir(directory):
    documents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = root + '/' + file
            lines = run_parser(full_path)
            doc = Document(full_path, lines)
            documents.append(doc)

    return documents


class Document:
    def __init__(self, path, lines):
        self.path = path
        self.lines = lines
        self.data = np.array([])
        self.target = np.array([])
        self.category = -1

