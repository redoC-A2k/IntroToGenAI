from langchain_openai import ChatOpenAI
import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import argparse

dotenv.load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# messages = [
#   SystemMessage("Translate the following from english to Italian"),
#   HumanMessage("hi! how are you?")
# ]

# print(model.invoke(messages).content)

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "Translate the following from english into {language}"), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Italian", "text": "hi! how are you?"})

response = model.invoke(prompt.to_messages())
print(response.content)