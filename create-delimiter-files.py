import os
import shutil
from shutil import copyfile
import sys
import re

text_files = os.listdir(sys.argv[1])
print(sys.argv[1])

for file in text_files:
    file_name = sys.argv[1] + file
    with open(file_name) as filehandle:
        lines = filehandle.readlines()
    non_empty = []
    for line in lines:

        if not (re.match(r'^\s*$', line)) :
            non_empty.append(line)

    f = open(file_name, 'w')
    f.writelines(non_empty)
    f.close()

for orig in text_files:
    original = sys.argv[1] + orig
    copy = "delimiters/" + orig
    copyfile(original, copy)


for file in text_files:
    file_name = "delimiters/" + file
    count = open(file_name).readlines()
    i = 1
    f = open(file_name, 'w+')
    range = len(count)
    print(range)
    for line in count :
        if i == (range -2) :
	    print("hello")
            break
        lineCount = str(i) + "," + str(i+2)
        f.write(lineCount)
        f.write("\n")
        i = i + 1
    f.close()


