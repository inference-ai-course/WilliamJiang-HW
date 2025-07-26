#Lecture 1 Prompt Engineering with Jupyter Notebook

import openai
import json

client = openai.OpenAI(api_key='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')
# Define available functions
def add_numbers(a, b):
    return a + b

def subtract_numbers(a, b):
    return a - b

def multiply_numbers(a, b):
    return a * b

# Function to get the model's response
def get_agent_response(user_prompt, model="gpt-4"):
    messages = [{"role": "user", "content": user_prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        functions=[
            {
                "name": "add_numbers",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "The first number"},
                        "b": {"type": "number", "description": "The second number"}
                    },
                    "required": ["a", "b"]
                }
            },
            {
                "name": "subtract_numbers",
                "description": "Subtract two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "The first number"},
                        "b": {"type": "number", "description": "The second number"}
                    },
                    "required": ["a", "b"]
                }
            },
            {
                "name": "multiply_numbers",
                "description": "Multiply two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "The first number"},
                        "b": {"type": "number", "description": "The second number"}
                    },
                    "required": ["a", "b"]
                }
            }
        ],
        temperature=0,
    )

    response_message = response.choices[0].message

    if response_message.function_call:
        function_name = response_message.function_call.name
        arguments = json.loads(response_message.function_call.arguments)
        if function_name == "add_numbers":
            result = add_numbers(**arguments)
        elif function_name == "subtract_numbers":
            result = subtract_numbers(**arguments)
        elif function_name == "multiply_numbers":
            result = multiply_numbers(**arguments)
        else:
            result = "Function not recognized."
        return result
    else:
        return response_message.content

# Example usage
user_prompt = "What is 5 times 8"
response = get_agent_response(user_prompt)
print(response)
