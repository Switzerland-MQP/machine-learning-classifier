""" Semantic validation takes place directly after parsing. The following things are checked:
1. Are the tag names valid (either made up of the categories, or 'data')  | invalid-tag-name
2. Are all <data> tags children of tags?  | no-lonely-data
3. Are there any tags nested in other tags?  | no-nested-tags

and converts to a new abstract syntax tree format (only difference is that the Tag class has an array of categories associatied with it)
"""

personal_categories = ['name','id-number', 'location', 'online-id', 'dob', 'phone', 'physical', 'physiological', 'professional', 'genetic', 'mental', 'economic', 'cultural', 'social']
sensitive_categories = ['criminal', 'origin', 'health', 'religion', 'political', 'philosophical', 'unions', 'sex-life', 'sex-orientation', 'biometric']
all_categories = personal_categories + sensitive_categories




from parser import Tag


def convert_ast(ast):
	pass
	
