# Sanitize the resume output
import os, re

resumedir = 'managerResumes/'

resumes = []
for filename in os.listdir(resumedir):
    resumeObject = open(resumedir + filename, 'r', errors='replace')
    resume = resumeObject.read()
    resume = resume[2:-2]
    newfile = open('managerResumesNew/'+filename, "a+")
    newfile.write(resume)
    newfile.close()
