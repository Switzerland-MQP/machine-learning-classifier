from html.parser import HTMLParser

class Tag():
	def __init__(self, start_pos):
		self.tag_open = True
		self.start_pos = start_pos
	
	def set_text(self, text, end_pos):
		if not self.tag_open:
			print("Error: tried to set text on a closed tag")
			quit()
		self.text = text
		self.close_pos = end_pos

	def close(self):
		self.tag_open = False

class Data(Tag):
	def __init__(self, start_pos):
		super().__init__(start_pos)

class Label(Tag):
	def __init__(self, start_pos, element_text):
		super().__init__(start_pos)
		self.tags = element_text.split('_')
		self.data_objects = []
		
	# Use tags to determine whether this label is personal or nonpersonal
	def get_information_type():
		return ""


class LabelParser(HTMLParser):
	def __init__(self):
		super().__init__()
		self.previous_pos = (0,0)
		self.document = []

	def handle_starttag(self, tag, attrs):
		if len(attrs) > 0:
			print("Error: tag has spaces in name")	
		print("Encountered a start tag:", tag)
		#if tag == 'data':
		#	#Check if we have an open Label
		#	if self.has_open_label():
		#		open_label = self.document[-1]
		#		if open_label.has_open_
		print(self.getpos())

	def handle_endtag(self, tag):
		print("Encountered an end tag:", tag)

	def handle_data(self, data):
		print("Encountered some data:", data)

if __name__ == "__main__":
	parser = LabelParser()
	document = """ <online-id_phone_name>Your contact:</online-id_phone_name>\n
		blah \nblah <name> my name is <data> griffin</data></name> \n
		and blah <name> my uncle's name is<data> dave</data></name> blah \n blah"""
	parser.feed(document)
