import os
import sys


if __name__ == "__main__":
    source = sys.argv[1]
    dest = sys.argv[2]
    for root, dirs, files in os.walk(source):
        path = root.split(os.sep)

        for file in files:
            print("File: ", file)
            # Call ToPlaintext function here
            destFileName = file + ".txt"
            destFileName = "{}/{}.txt".format(dest, file)
            print("Dest File Name: ", destFileName)

            destFile = open(destFileName, "w")
            destFile.close()
