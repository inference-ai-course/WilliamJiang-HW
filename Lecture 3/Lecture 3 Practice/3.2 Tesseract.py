import pytesseract
from PIL import Image

# Load and preprocess image (convert to grayscale)
image = Image.open("D:\Inference Ai Stuff\Lecture 2\Lecture 2 Homework\Joker poster.jpg").convert("L")  # grayscale
text = pytesseract.image_to_string(image)

print("📄 Tesseract OCR Output (first 500 chars):")
print(text[:500])
