from dotenv import load_dotenv
from openai import OpenAI
import json

if load_dotenv():
    print("Successfully loaded api key")

# ---Completions  API --- 
# API Q1

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print(f'''
    Model:{response.model}
    Usage: {response.usage}
    message: {response.choices[0].message.content}
    ''')

# API Q2


response_07 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}],
    temperature=0.7

)
response_10 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}],
    temperature=1.0
)

response_15 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}],
    temperature=1.5
)

print(f'''
    Temp 0.7: {response_07.choices[0].message.content}
    Temp 1.0: {response_10.choices[0].message.content}
    Temp 1.5: {response_15.choices[0].message.content}
    ''')
#The higher the temperture the more creative nonprecise language was used. For the most consistent respinses the lowest temperture would be the most reliable


#API Q3
response_10 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}],
    temperature=1.0
)

response_t1_n_3 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    temperature=1.0,
    n=3
)

for i in range(0,3):
    print(response_t1_n_3.choices[i].message.content)

# #API 4
response_long = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "explain Albert Einstein's greatest theories(minimun 15) in detail"}],
    max_tokens=15
)
print(response_long.choices[0].message.content)

#it stopped the response mid sentence. The token limit may be used to make sure end users are not overutilizing resources in queries 


# ---System Question---
#Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
    ],
    n=3,
    temperature=1.0
)
print(response.choices[0].message.content)

response_professor = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
    {"role": "system", "content": "You are a nobel proze winning physicist who is explaining python to graduate students."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
    ],
    n=3,
    temperature=1.0
)
print(response_professor.choices[0].message.content)
#the main difference is the examples that were used with the professor system prompt is the exaample was a bit more advanced. Overall the setiment was similar

#Q2

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
    ],
    n=3,
    temperature=1.0
)
print(response.choices[0].message.content)
#because all of that information was in the orgingal message context so they are all apllied when responding to the user's final question
#the name is inside of the orignal state for the message response


# ---Prompt Engineering   --- 

#Q1

def get_completion(model="gpt-4o-mini", temperature=0, input='', improvement=''):
    prompt = f"""
    Analyze the customer review below and extract:
    1. The overall sentiment 
    2. Any specific issues mentioned

    Review:
    '''
    {input}
    '''
    {improvement}
    Format your response as:

    Sentiment: <sentiment>
    Issues: <list of issues or "none">
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for review in reviews:

    response=get_completion(input=review)
    print(response)

#Q2

example='''Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed
'''
print('With Example||Oneshot')
for review in reviews:
    response=get_completion(input=review, improvement=example)
    print(response)

#adding one example did not change the responses at all they are nearly identical
#Q3
example_few='''
    Example:
    Review1: "The customer service team resolved my issue within minutes. Truly outstanding support experience!" 
    Sentiment1: positive 
    Review2: "The product stopped working after two days and the return process was a complete nightmare."
    Sentiment2: negative 
    Review3: "Fast shipping but the item arrived damaged."
    Sentiment3: mixed
'''

print('With Example||Fewshot')
for review in reviews:
    response=get_completion(input=review, improvement=example_few)
    print(response)
#so there still is not much of a difference, there have been other instances where the more examples there are the response from the llm is better

#Q4


def get_completion_delta(model="gpt-4o-mini", temperature=0):
    prompt = """
    Show your step-by-step reasoning, then give the final answer on its own line labelled: Final answer: <value>

    Problem:A data engineer earns $85,000 per year. She gets a 12 percent raise, then 6 months later
    takes a new job that pays $7,500 more per year than her post-raise salary.
    What is her final annual salary?
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content


response_problem = get_completion_delta()
print(response_problem)

#it builds trust in the model, allows you to iteriate better on the prompts and it helps the model break down larger queries into smaller more digestable step
#this means ther are less logical leaps and more accurate answers

# Q5


prompt_json= '''Analyze the sentiment of this customer review and respond only with valid JSON.
Return keys: sentiment, confidence (0–1), brief_reason.

Review:I've been using this tool for three months. It handles large datasets well, 
but the UI is clunky and the export options are limited
'''

def get_completion_json(model="gpt-4o-mini", temperature=0, prompt_json=""):
    prompt = prompt_json
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    response= response.choices[0].message.content
    try:
        print('hey')
        return json.loads(response)
    except json.JSONDecodeError:
        print("JSON Invalid")
        print(response)

json_response= get_completion_json(prompt_json=prompt_json)
print(json_response)

#Q5
user_text = '''First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice.
'''

prompt_1 = f"""
    You will be given text inside triple backticks.
    If it contains step-by-step instructions, rewrite them as a numbered list.
    If it does not contain instructions, respond with exactly: "No steps provided."

    ```{user_text}```
    """
def get_completion_delim(model="gpt-4o-mini", temperature=0, prompt_delim=""):

    prompt=prompt_delim
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    response= response.choices[0].message.content
    return response


response_delim= get_completion_delim(prompt_delim=prompt_1)
print(response_delim)

non_instruction_text = "The sun sets earlier in winter because of the tilt of the Earth's axis \
relative to its orbit around the sun."

prompt2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{non_instruction_text}```
"""

response_delim_2= get_completion_delim(prompt_delim=prompt2)
print(response_delim_2)

#it helps seperate the context and instructions from the actual input so the model understand what  directions its following vs what iput to act on
ollama_prompt="Explain what a large language model is in two sentences."

# Q6
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": ollama_prompt}]
)
print(response.choices[0].message.content)

#Ollama response
#Thinking...
# Okay, the user wants me to explain what a large language model is in two sentences. Let me start 
# by recalling what I know about them. First, I remember that large language models are artificial 
# intelligence models trained on vast amounts of text. They're used in various fields like 
# language processing, customer service, etc.

# Now, how to structure two sentences? The first sentence should introduce the main points: the 
# model's purpose and the training data. The second sentence needs to give a brief example or a 
# function. Maybe mention applications like customer support or language translation. Wait, I 
# should make sure it's concise and clear. Let me check if I'm including all necessary elements. 
# Also, avoid technical jargon but keep it understandable. Alright, that should cover it.
# ...done thinking.

# A large language model is an artificial intelligence system trained on vast amounts of text, 
# enabling it to understand and generate human-like language. It's used in various applications 
# like language translation or customer service to process and respond to human queries.


#ollama automatical adds the thinking, but it also takes a bit longer, likely based on the power of my laptop vs the speed of an api
#other wise the response sematically are very similar