#import numpy as np
from run_parser import run_parser
from os import listdir
from os.path import isfile, join
from to_lines import Line

parent_dir = '../TEXTDATA/LINE_BY_LINE'
personal_dir = parent_dir + '/PERSONAL_DATA/html-tagged'
sensitive_dir = parent_dir + '/SENSITIVE_DATA/html-tagged'

def get_files_in_dir(path):
	return [join(path, f) for f in listdir(path) if isfile(join(path, f))]

def count_lines(files):
	files_processed = 0
	error_files = 0

	total_lines = 0
	total_lines_tagged = 0
	category_totals = {}
	for f in files:
		try:
			lines = run_parser(f)
			files_processed += 1
		except Exception as e:		
			error_files += 1
			print(f + " --> " + str(e))
			continue			

		#print("File " + f + " has " + str(len(lines)) + " lines")

		for line in lines:
			total_lines += 1
			if line.categories or line.context:
				total_lines_tagged += 1
			for category in line.categories:
				if category not in category_totals.keys():
					category_totals[category] = 0
				category_totals[category] += 1
	print("Processed " + str(files_processed) + " files, skipping " + str(error_files) + " files due to errors")
	print(" ============================")
	print("Total lines:", total_lines)
	print("Total lines with tags:", total_lines_tagged)
	print("====== Category totals ======")
	for category in category_totals.keys():
		print(category + " => " + str( category_totals[category] ))


files = get_files_in_dir(personal_dir)
files = files + get_files_in_dir(sensitive_dir)

count_lines(files)

#print(" ================================================" )
#files = get_files_in_dir(parent_dir + "/PERSONAL_DATA/")
#files = files + get_files_in_dir(parent_dir + "/SENSITIVE_DATA/")

#count_lines(files)

		