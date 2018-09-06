import numpy as np
import os
from parser.run_parser import run_parser
from sklearn.model_selection import train_test_split

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
    all_documents = []
    for d in directories:
        all_documents += load_dir_custom(d)
    return all_documents


def document_test_train_split(documents, test_size):
    document_labels = get_documents_labels(documents)
    doc_train, doc_test, _, _ = train_test_split(
        documents, document_labels, test_size=test_size, shuffle=True
    )
    return doc_train, doc_test


def load_dir_custom(directory):
    documents = read_dir(directory)
    fill_docs(documents)
    return documents


def fill_docs(documents):
    for doc in documents:
        data = np.array([])
        targets = np.array([])
        for line in doc.lines:
            data = np.append(data, [line.text])
            targets = np.append(targets, [convert_categories(line.categories)])
        doc.data = data
        doc.targets = targets
        doc.category = classify_doc(targets)


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


def get_documents_labels(documents):
    targets = []
    for doc in documents:
        targets.append(doc.category)
    return targets


def convert_docs_to_lines(documents):
    targets = np.array([])
    data = np.array([])
    for doc in documents:
        targets = np.append(targets, doc.targets)
        data = np.append(data, doc.data)
    return data, targets


class Document:
    def __init__(self, path, lines):
        self.path = path
        self.lines = lines
        self.data = np.array([])
        self.targets = np.array([])
        self.category = -1
