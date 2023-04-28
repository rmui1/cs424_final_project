import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="I'm feeling hungry. There's a burger, table, fridge. Which item is best for me? Answer with one word, all lowercase, and no punctuation. Then, rate the object in terms of its usefulness to me at the moment, from 1 to 10. Explain. Phrase your response in the form: item;number;explanation",
  temperature=0.7,
  max_tokens=2048,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response['choices'][0]['text'].strip())
