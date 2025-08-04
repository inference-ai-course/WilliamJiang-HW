
from PIL import Image
import pytesseract #pip install pytesseract first


# Load an image using Pillow (PIL)
image = Image.open('D:\\Inference Ai Stuff\\Lecture 2\\Lecture 2 Homework\\Joker poster.jpg')
# print (image)
# print (image.size)
# image.thumbnail((100,100))
# image.show()

# Perform OCR on the image
text = pytesseract.image_to_string(image)

print(text)