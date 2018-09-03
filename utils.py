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
    all_data = []
    all_targets = []
    for d in directories:
        data, target = load_dir_custom(d)
        all_data = all_data + data
        all_targets = all_targets + target
    return all_data, all_targets


def load_dir_custom(directory):
    lines = read_dir(directory)

    data = np.array([])
    target = np.array([])
    for l in lines:
        data = np.append(data, [l.text])
        target = np.append(target, [convert_categories(l.categories)])
    return data, target


def convert_categories(categories):
    for c in sensitive_categories:
        if c in categories:
            return 2
    for c in personal_categories:
        if c in categories:
            return 1
    return 0


def read_dir(directory):
    documents = {}
    all_lines = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = root + '/' + file
            lines = run_parser(full_path)
            documents[full_path] = lines
            all_lines = all_lines + lines

    return all_lines
