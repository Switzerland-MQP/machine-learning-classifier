from parser import LabelParser
from semantic_analysis import convert_ast

parser = LabelParser()
document = """ <online-id_phone_name>Your contact:</online-id_phone_name>\n
		blah \nblah <name> my name is <data> griffin</data></name> \n
		and blah <name> my uncle's name is<data> dave</data></name> blah \n blah"""
ast = parser.parse(document)

print("Input :\n ", document)
print(ast)

ast = convert_ast(ast)


