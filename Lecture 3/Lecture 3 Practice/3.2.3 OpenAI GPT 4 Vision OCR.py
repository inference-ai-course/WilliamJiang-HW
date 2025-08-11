import base64
import requests

def vision_extract(b64_image, prompt, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ]}
        ],
        "max_tokens": 3000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

# Load image and run GPT-4o OCR
with open("D:/Inference Ai Stuff/Lecture 3/Lecture 3 Practice/test.png", "rb") as f:
    b64_img = base64.b64encode(f.read()).decode("utf-8")

# Use your actual API key here
result = vision_extract(b64_img, "Extract all the readable text from this document.", api_key="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")
print(result["choices"][0]["message"]["content"])