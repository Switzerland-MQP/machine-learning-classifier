import os
import uuid
import sys

from plaintext_converter import to_plaintext


def getDestFilename(dest, file):
    destFileName = "{}/{}.txt".format(dest, file)
    if os.path.exists(destFileName):
        destFileName = "{}/{}-{}.txt".format(dest, file, str(uuid.uuid4()))
    return destFileName


def valid(file):
    whitelist = [
        ".csv", ".doc", ".docx", ".eml",
        ".epub", ".gif", ".htm", ".html",
        ".jpeg", ".jpg", ".json", ".log",
        ".mp3", ".msg", ".odt", ".ogg", ".pdf",
        ".png", ".pptx", ".ps", ".psv", ".rtf",
        ".tff", ".tif", ".tiff", ".tsv", ".txt",
        ".wav", ".xls", ".xlsx"
    ]
    _, file_extension = os.path.splitext(file)
    return file_extension in whitelist


def convert_directory(source, dest):
    if not os.path.isdir(source):
        print("Please give a directory as the first argument")
        os.exit()

    if not os.path.isdir(dest):
        os.makedirs(dest)

    original_file_names = []
    failed_files = []
    for root, dirs, files in os.walk(source):
        for file in files:
            #  print("File: ", file)

            if not valid(file):
                continue

            full_file_name = root + "/" + file

            try:
                text = to_plaintext.to_plaintext(full_file_name)
            except Exception as e:
                logfile = open("errors.log", "a")
                logfile.write("Couldn't convert file: {} with message: {}\n".format(
                    full_file_name, e
                ))
                logfile.close()
                failed_files.append(full_file_name)
                continue

            destFileName = getDestFilename(dest, file)
            destFile = open(destFileName, "wb")
            destFile.write(text)
            destFile.close()
            original_file_names += file
    print(f"Failed files: {failed_files}")
    return failed_files
