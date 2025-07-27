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


prompt = "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Explain your reasoning."
response = get_completion(prompt)
print(response)

print("\n" + "---------------------------------------------------" + "\n")

prompt = "If a water tank can hold 1000 liters of water, and a pump can fill it in 10 minutes, how long will it take to fill a 500 liter tank? Answer in a step by step manner."
response = get_completion(prompt)
print(response)