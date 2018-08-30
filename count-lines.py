import os
import shutil
from shutil import copyfile
import sys
import re


text_files = os.listdir(sys.argv[1])
print(sys.argv[1])

num_lines = 0
for file in text_files:
    file_name = sys.argv[1] + file
    if os.path.isfile(file_name) :
        with open(file_name) as filehandle:
            lines = filehandle.readlines()
        non_empty = []
        for line in lines:

            if not (re.match(r'^\s*$', line)) :
                non_empty.append(line)
        num_lines = num_lines + len(non_empty)


print("Total number of lines: " + str(num_lines))


num_data = 0
for file in text_files:
    file_name = sys.argv[1] + file
    if os.path.isfile(file_name) :
        with open(file_name) as filehandle:
            lines = filehandle.readlines()
        for line in lines:
            regexp = re.compile(r'<data>')
            matches = regexp.findall(line)
            num_data = num_data + len(matches)


print("Total number of tags: " + str(num_data))
