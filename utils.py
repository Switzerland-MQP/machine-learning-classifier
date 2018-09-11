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


def load_dirs_custom(directories, individual=False):
    all_documents = []
    for d in directories:
        all_documents += load_dir_custom(d, individual)
    return all_documents


def document_test_train_split(documents, test_size):
    document_labels = get_documents_labels(documents)
    doc_train, doc_test, _, _ = train_test_split(
        documents, document_labels, test_size=test_size, shuffle=True
    )
    return doc_train, doc_test


def load_dir_custom(directory, individual):
    documents = read_dir(directory)
    fill_docs(documents, individual)
    return documents


def fill_docs(documents, individual=False):
    for doc in documents:
        data = np.array([])
        targets = np.array([])
        contexts = np.array([])
        for line in doc.lines:
            data = np.append(data, [line.text])
            targets = np.append(
                targets,
                [convert_categories(line.categories, individual)]
            )
            contexts = np.append(
                contexts,
                [convert_categories(line.context, individual)]
            )
        doc.data = data
        doc.contexts = contexts
        doc.targets = targets
        doc.category = classify_doc(targets)


def classify_doc(target_array):
    return max(target_array)


def convert_categories(categories, individual):
    if individual:
        return convert_categories_individual(categories)
    else:
        return convert_categories_buckets(categories)


def convert_categories_individual(categories):
    category_list = ['phone']
    for c in range(len(category_list)):
        if category_list[c] in categories:
            return c+1
    return 0


def convert_categories_buckets(categories):
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


def convert_docs_to_lines(documents, context=False):
    targets = np.array([])
    data = np.array([])
    contexts = np.array([])
    for doc in documents:
        targets = np.append(targets, doc.targets)
        data = np.append(data, doc.data)
        contexts = np.append(contexts, doc.contexts)

    if context:
        return data, targets, contexts
    else:
        return data, targets


def n_gram_documents_range(docs, low, high):
    for d in docs:
        n_gram_document_range(d, low, high)
    return docs


def n_gram_document_range(doc, low, high):
    all_data = np.array([])
    all_target = np.array([])
    for i in range(low, high+1):
        data, targets = n_grams(doc.data, doc.targets, i)
        all_data = np.append(all_data, data)
        all_target = np.append(all_target, targets)

    doc.data = all_data
    doc.targets = all_target


def n_gram_documents(docs, n):
    for d in docs:
        n_gram_document(d, n)
    return docs


def n_gram_document(doc, n):
    data, targets = n_grams(doc.data, doc.targets, n)
    doc.data = data
    doc.targets = targets

    return doc


def n_grams(data_array, target_array, n):
    grams = np.array([])
    targets = np.array([])
    for i in range(len(data_array) - n + 1):
        new_str = '\n'.join(data_array[i:i+n])
        grams = np.append(grams, [new_str])
        targets = np.append(targets, [max(target_array[i:i+n])])
    return grams, targets


class Document:
    def __init__(self, path, lines):
        self.path = path
        self.lines = lines
        self.data = np.array([])
        self.targets = np.array([])
        self.contexts = np.array([])
        self.category = -1
