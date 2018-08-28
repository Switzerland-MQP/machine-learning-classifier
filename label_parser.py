from html.parser import HTMLParser

personal_categories = ['name','id-number', 'location', 'online-id', 'dob', 'phone', 'physical', 'physiological', 'professional', 'genetic', 'mental', 'economic', 'cultural', 'social']
sensitive_categories = ['criminal', 'origin', 'health', 'religion', 'political', 'philosophical', 'unions', 'sex-life', 'sex-orientation', 'biometric']
all_categories = personal_categories + sensitive_categories

def get_index_of_nth_line(string, n):
	n -= 1
	i = 0
	counter = 0
	while n > 0:
		if string[counter] == '\n':
			n -= 1
			counter += 1
			continue
		i += 1
		counter += 1	
	return i

def pos_splice(raw_document, start_pos, end_pos):
	start_line, start_index = start_pos
	end_line, end_index = end_pos	

	start = get_index_of_nth_line(raw_document, start_line) + start_index
	end = get_index_of_nth_line(raw_document, end_line) + end_index

	return raw_document[start:end]


def pos_slice_wrapper(raw_document, start_pos, end_pos):
	print("Slicing:", start_pos, " --> ", end_pos, "\n of doc: \n", raw_document, "||")
	print(pos_splice(raw_document, start_pos, end_pos))


class LabelParser(HTMLParser):
	def __init__(self):
		super().__init__()
		self.raw_document = ''
		self.previous_pos = (0,0)
		self.document = []
		self.status = 'text'
		self.current_labels = []

	def feed(self, raw_document):
		self.raw_document = raw_document
		super().feed(raw_document)
	
	def tag_is_valid(self, delimited_tag):
		tags = delimited_tag.split('_')
		return all([tag in all_categories for tag in tags])

	def label_lines(self, start_pos, end_pos, labels):
		pass

	def handle_starttag(self, tag, attrs):
		if len(attrs) > 0:
			raise Exception("Error: tag has spaces in name")
	
		print(self.getpos(), "Encountered a start tag - Current status: ", self.status, " | : ", tag)
		
		if self.tag_is_valid(tag):
			if self.status != 'text':
				raise Exception("Error: open tag " + tag + " encountered without closing previous tag")

			self.label_lines(self.previous_pos, self.getpos(), ['not personal'])
			self.current_labels = tag.split('_')
			self.status = 'label'
		elif tag == 'data':
			if self.status != 'label':
				raise Exception("Error: open data tag encounted outside of a label")
			if len(self.current_labels) < 1:
				raise Exception("Error: open data tag encountered but no labels were available to apply")			

			self.label_lines(self.previous_pos, self.getpos(), [label + '-context' for label in self.current_labels])
			self.status = 'data'
		else:
			raise Exception("Error: " + tag + " is not a valid category")


	def handle_endtag(self, tag):
		print(self.getpos(), "Encountered an end tag - Status:", self.status, "| :", tag)
		if self.tag_is_valid(tag):
			if self.status != 'label':
				raise Exception("Error: Close tag ", tag, " encountered without a matching open tag")
			if self.current_labels != tag.split('_'):
				raise Exception("Error: Close tag ", tag, " does not match open tag ", self.current_labels.join('_'))

			self.label_lines(self.previous_pos, self.getpos(), [label + '-context' for label in self.current_labels])
			self.current_labels = []
			self.status = 'text'
		elif tag == 'data':
			if self.status != 'data':
				raise Exception("Error: </data> close tag encountered without a matching open data tag")
			
			self.label_lines(self.previous_pos, self.getpos(), self.current_labels)
			self.status = 'label'
		else:
			raise Exception("Error: close tag " + tag + " is not a valid tag")

	def handle_data(self, data):
		print(self.getpos(), "Encountered some data - Status:", self.status, "| :", data)


if __name__ == "__main__":
	#parser = LabelParser()
	#document = """ <online-id_phone_name>Your contact:</online-id_phone_name>\n
	#	blah \nblah <name> my name is <data> griffin</data></name> \n
	#	and blah <name> my uncle's name is<data> dave</data></name> blah \n blah"""
	#parser.feed(document)

	document = """0123456789\n0123456789\n"""
	pos_slice_wrapper(document, (1, 0), (1, 5))






