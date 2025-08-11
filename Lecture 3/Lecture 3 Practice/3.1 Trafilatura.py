import trafilatura
import requests

# Example: An arXiv paper abstract page
url = "https://arxiv.org/abs/2404.00001"

# Step 1: Fetch raw HTML
response = requests.get(url)
html = response.text

# Step 2: Use Trafilatura to extract clean text
downloaded_text = trafilatura.extract(html, include_comments=False, include_tables=False)

# Step 3: Display the result
print("📄 Extracted Text Preview:\n")
print(downloaded_text[:1000])  # Show first 1000 characters