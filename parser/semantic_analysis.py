""" Semantic validation takes place directly after parsing. The following things are checked:
1. Are the tag names valid (either made up of the categories, or 'data')  | invalid-tag-name
2. Are any top level <data> tags?  | no-lonely-data
3. Are there any tags nested in other tags?  | no-nested-tags

and converts to a new abstract syntax tree format (only difference is that the Tag class has an array of categories associatied with it)
"""

personal_categories = ['name','id-number', 'location', 'online-id', 'dob', 'phone', 'physical', 'physiological', 'professional', 'genetic', 'mental', 'economic', 'cultural', 'social']
sensitive_categories = ['criminal', 'origin', 'health', 'religion', 'political', 'philosophical', 'unions', 'sex-life', 'sex-orientation', 'biometric']
all_categories = personal_categories + sensitive_categories


from parser import Tag

	
def validate_ast(ast):
	for element in ast:
		print(type(element))
			
		if type(element) is Tag:
			if not are_valid_categories(element.name.split("_")) and not element.name == 'data':
				raise Exception("Semantic Error: ", element.name, " is not a valid category or data label")
		
			element.set_categories(element.name.split("_"))
			element.set_name("Label")

			if element.name == 'data':
				raise Exception("Semantic Error: top-level data tag found")

			return validate_children(element.children)
		else:
			print("fffff")

def validate_children(children):
	for element in children:
		if type(element) is str:
			return
		elif type(element) is Tag:
			if has_valid_tag_name(element):
				raise Exception("Semantic Error: label ", element.name, " cannot be nested under another label")
			elif not element.name == 'data':
				raise Exception("Semantic Error: label ", element.name, " is not recognized and can not be nested under another label")	

def are_valid_categories(categories):
	for category in categories:
		if category not in all_categories:
			return False
	return True

