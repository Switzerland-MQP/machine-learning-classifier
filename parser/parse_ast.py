from html.parser import HTMLParser

personal_categories = ['name','id-number', 'location', 'online-id', 'dob', 'phone', 'physical', 'physiological', 'professional', 'genetic', 'mental', 'economic', 'cultural', 'social']
sensitive_categories = ['criminal', 'origin', 'health', 'religion', 'political', 'philosophical', 'unions', 'sex-life', 'sex-orientation', 'biometric']
all_categories = personal_categories + sensitive_categories


class Tag:
	def __init__(self, name):
		self.name = name
		self.children = []
		self.is_open = True
	
	def __repr__(self):
		return self.name + "(" + str(self.children) + ")"

	def name(self, name):
		self.name = name

	def close(self):
		self.is_open = False

class LabelParser(HTMLParser):
	def __init__(self):
		super().__init__()
		self.raw_document = ''
		self.document = []
		self.verbosity = 0

	def parse(self, raw_document):
		self.raw_document = raw_document
		super().feed(raw_document)
		return self.document

	def handle_starttag(self, tag_name, attrs):
		if self.verbosity > 1:
			print(self.getpos(), "Encountered a start tag - Current status| : ", tag_name)
		
		if len(attrs) > 0:
			raise Exception("Parse Error: tag has spaces in name")
	
		self.handle_starttag_helper(self.document, tag_name)	

	
	def handle_starttag_helper(self, on_list, tag_name):
		if len(on_list) < 1:
			on_list.append(Tag(tag_name))
			return		
		last_expression = on_list[-1]
	
		if type(last_expression) is str:
			on_list.append(Tag(tag_name))
		elif type(last_expression) is Tag:
			if not last_expression.is_open:
				on_list.append(Tag(tag_name))
			else: #last_expression is open
				self.handle_starttag_helper(last_expression.children, tag_name)


	def handle_endtag(self, tag_name):
		if self.verbosity > 1:
			print(self.getpos(), "Encountered an end tag| :", tag_name)
		
		self.handle_endtag_helper(self.document, tag_name)

	
	def handle_endtag_helper(self, on_list, tag_name):
		if len(on_list) < 1:
			raise Exception("Parse Error: close tag mismatch:", tag_name)
		last_expression = on_list[-1]

		if type(last_expression) is str:
			raise Exception("Parse Error: encountered close tag ", tag_name, " without a matching open tag")
		elif type(last_expression) is Tag:
			if not last_expression.is_open:
				raise Exception("Parse Error: encountered close tag ", tag_name, " without an open tag")
			else: # last_expression is open
				if not last_expression.name == tag_name:
					self.handle_endtag_helper(last_expression.children, tag_name)	
				else:
					last_expression.close()


	def handle_data(self, text):
		if self.verbosity > 1:
			print(self.getpos(), "Encountered some text | :", text)
	
		self.handle_data_helper(self.document, text)

	
	def handle_data_helper(self, on_list, text):
		if len(on_list) < 1:
			on_list.append(text)
			return
		last_expression = on_list[-1]
		
		if type(last_expression) is str:
			raise Exception("Parse internal Error: encountered text directly after text")
		elif type(last_expression) is Tag:
			if not last_expression.is_open:
				on_list.append(text)
			else:
				#last expression is open
				self.handle_data_helper(last_expression.children, text)


if __name__ == "__main__":
	parser = LabelParser()
	document = """ <online-id_phone_name>Your contact:</online-id_phone_name>\n
		blah \nblah <name> my name is <data> griffin</data></name> \n
		and blah <name> my uncle's name is<data> dave</data></name> blah \n blah"""
	ast = parser.parse(document)

	print("Input :\n ", document)
	print(ast)

	#document = """0123456789\n0123456789\n"""
	#pos_slice_wrapper(document, (1, 0), (1, 5))






