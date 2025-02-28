from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class Person(BaseModel):
    """Information about a person"""
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(default=None, description="The color of the person's hair if known")
    height_in_meters: Optional[str] = Field(default=None, description="Height measured in meters")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert extraction algorithm. Only extract the relevant information from the text. If you do not know the value of an attribute asked to extract return null for the attribute's value"),
    ("human", "{text}")
])

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b").with_structured_output(schema=Person)
# llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(schema=Person)

text = "Alan Smith is 6 feet tall and has blond hair."
prompt = prompt_template.invoke({"text": text})
# result = llm.invoke(prompt)

# print(result)

# Multiple entities

class Data(BaseModel):
    """Extracted data about people"""
    people: List[Person]

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash").with_structured_output(schema=Data)
llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(schema=Data)
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
prompt = prompt_template.invoke({"text": text})
result = llm.invoke(prompt)
print(result)