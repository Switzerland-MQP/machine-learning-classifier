from html.parser import HTMLParser

personal_categories = ['name','id-number', 'location', 'online-id', 'dob', 'phone', 'physical', 'physiological', 'professional', 'genetic', 'mental', 'economic', 'cultural', 'social']
sensitive_categories = ['criminal', 'origin', 'health', 'religion', 'political', 'philosophical', 'unions', 'sex-life', 'sex-orientation', 'biometric']
all_categories = personal_categories + sensitive_categories


class Tag:
	def __init__(self, name):
		self.name = name
		self.children = []
		self.is_open = True

	def name(self, name):
		self.name = name

	def add_child(self, child)
		self.children = self.children + [child]

	def close(self):
		self.is_open = False

class LabelParser(HTMLParser):
	def __init__(self):
		super().__init__()
		self.raw_document = ''
		self.document = []
		self.status = 'text'

	def feed(self, raw_document):
		self.raw_document = raw_document
		super().feed(raw_document)
	
	def handle_starttag(self, tag_name, attrs):
		print(self.getpos(), "Encountered a start tag - Current status| : ", tag_name)
		
		if len(attrs) > 0:
			raise Exception("Parse Error: tag has spaces in name")
		
		if len(self.document) < 1:
			self.document.append(Tag(tag_name))
			return		
		last_expression = self.document[-1]
	
		if type(last_expression) is str:
			self.document.append(Tag(tag_name))
		elif type(last_expression) is Tag:
			if not last_expression.is_open:
				self.document.append(Tag(tag_name))
			else: #last_expression is open
				last_expression.add_child(Tag(tag_name))
				

	def handle_endtag(self, tag_name):
		print(self.getpos(), "Encountered an end tag| :", tag_name)
		
		if len(self.document) < 1:
			raise Exception("Parse Error: encountered close tag at start of document")
		last_expression = self.document[-1]

		if type(last_expression) is str:
			raise Exception("Parse Error: encountered close tag ", tag_name, " without a matching open tag")
		elif type(last_expression) is Tag:
			if not last_expression.is_open:
				raise Exception("Parse Error: encountered close tag ", tag_name, " without an open tag")
			else: # last_expression is open
				if not last_expression.name == tag_name:
					raise Exception("Parse Error: mismatch close tag ", tag_name, " for open tag ", last_expression.name)
				else:
					last_expression.close()

	def handle_data(self, text):
		print(self.getpos(), "Encountered some text | :", text)

		if len(self.document) < 1:
			self.document.append(text)
			return
		last_expression = self.document[-1]
		
		if type(last_expression) is str:
			raise Exception("Parse internal Error: encountered text directly after text")
		elif type(last_expression) is Tag:
			if not last_expression.is_open:
				self.document.append(text)
			else:
				#last expression is open
				last_expression.add_child(text)

	def handle_data_helper(self, on_list, text):
		pass

if __name__ == "__main__":
	#parser = LabelParser()
	#document = """ <online-id_phone_name>Your contact:</online-id_phone_name>\n
	#	blah \nblah <name> my name is <data> griffin</data></name> \n
	#	and blah <name> my uncle's name is<data> dave</data></name> blah \n blah"""
	#parser.feed(document)

	document = """0123456789\n0123456789\n"""
	pos_slice_wrapper(document, (1, 0), (1, 5))






