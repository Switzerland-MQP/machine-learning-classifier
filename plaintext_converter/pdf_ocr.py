import sys
from pdf2image import convert_from_path
from pytesseract import image_to_string
from PIL import Image

images = convert_from_path(sys.argv[1])

text = ""
for image in images:
	image_text = image_to_string(image)
	text += image_text

print(text)
