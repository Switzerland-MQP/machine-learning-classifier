import os
import sys

from to_plaintext import to_plaintext

if __name__ == "__main__":
    source = sys.argv[1]
    dest = sys.argv[2]
    whitelist = [
        ".csv", ".doc", ".docx", ".eml",
        ".epub", ".gif", ".htm", ".html",
        ".jpeg", ".jpg", ".json", ".log",
        ".mp3", ".msg", ".odt", ".ogg", ".pdf",
        ".png", ".pptx", ".ps", ".psv", ".rtf",
        ".tff", ".tif", ".tiff", ".tsv", ".txt",
        ".wav", ".xls", ".xlsx"
    ]
    for root, dirs, files in os.walk(source):
        path = root.split(os.sep)

        for file in files:
            print("File: ", file)

            _, file_extension = os.path.splitext(file)
            if file_extension not in whitelist:
                continue

            full_file_name = root + "/" + file
            try:
                text = to_plaintext(full_file_name)
            except:
                logfile = open("errors.log", "w")
                logfile.write("Couldn't convert file: {}".format(full_file_name))
                logfile.close()

            destFileName = file + ".txt"
            destFileName = "{}/{}.txt".format(dest, file)
            print("Dest File Name: ", destFileName)

            destFile = open(destFileName, "wb")
            destFile.write(text)
            destFile.close()



