#Lecture 1 Prompt Engineering with Jupyter Notebook

import openai

# Initialize the OpenAI client
client = openai.OpenAI(api_key='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')

def get_completion(prompt, model="gpt-4o-mini"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

text = """
John Doe, a 29-year-old software engineer from San Francisco, recently joined OpenAI as a research scientist.
"""

prompt = f"Extract the name and occupation from the following text:\n{text}"
response = get_completion(prompt)
print(response)


text = """
John Doe, a 29-year-old software engineer from San Francisco, recently joined OpenAI as a research scientist.
"""

prompt = f"Extract the age and location from the following text:\n{text}"
response = get_completion(prompt)
print(response)
