from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
load_dotenv()

# result = llm.invoke("What is the square root of 16?")
# print(result)

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage:
Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
""")

class Classification(BaseModel):
    # sentiment: str = Field(..., enum=["happy","neutral","sad"], description="The sentiment of the text.")
    # aggressiveness: int = Field(..., enum=[1,2,3,4,5], description="describes how aggressive the statement is, the higher the number the more aggressive")
    # language: str = Field(..., enum=["spanish", "english", "french", "german", "italian"], description="The language the text is written in")
    sentiment: str = Field(..., enum=["happy","neutral","sad"], description="The sentiment of the text.")
    aggressiveness: int = Field(..., description="describes how aggressive the statement is, the higher the number the more aggressive")
    language: str = Field(..., enum=["english","spanish", "french", "german", "italian"], description="The language the text is written in")

llm = ChatGoogleGenerativeAI(temperature=0, model = "gemini-1.5-flash").with_structured_output(Classification)

input = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": input})
response = llm.invoke(prompt)
print(response)