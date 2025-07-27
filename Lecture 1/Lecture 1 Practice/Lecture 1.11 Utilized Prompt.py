#Lecture 1 Prompt Engineering with Jupyter Notebook

import openai

# Initialize the OpenAI client
client = openai.OpenAI(api_key='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')

messages = [
    {"role": "system", "content": "You are a helpful assistant knowledgeable in history that is funny and sarcastic."},
    {"role": "user", "content": "Who was the 20th president of the United States?"},
    {"role": "assistant", "content": "George Washington was the first president of the United States."},
    {"role": "user", "content": "When did they take office?"}
]

# Get the model's response
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0.7,
)

# Output the assistant's reply
print(response.choices[0].message.content)