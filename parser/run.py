from parser import LabelParser
from semantic_analysis import validate_ast
from to_lines import flag_lines
import sys

if len(sys.argv) > 1:
	document = " ".join(sys.argv[1:])
	print("Input: " + document)

	parser = LabelParser()
	ast = parser.parse(document)
	validate_ast(ast)
	print("AST: ", ast)

	lines = flag_lines(ast)
	print("Lines:", lines)
		
	quit()

parser = LabelParser()
#document = """ <online-id_phone_name>Your contact:</online-id_phone_name>\n
#		blah \nblah <name> my name is <data> griffin</data></name> \n
#		and blah <name> my uncle's name is<data> dave</data></name> blah \n blah"""

document = """ te\nst <phone> pho\nne: <data> 978 \n 345 1234 <data> test test""" 

ast = parser.parse(document)

print("Input :\n ", document)
print(ast)

validate_ast(ast)
print("||||||||||||||||||||||||")
print(ast)

flagged_lines = flag_lines(ast)
print("Flagged lines:\n", flagged_lines)

