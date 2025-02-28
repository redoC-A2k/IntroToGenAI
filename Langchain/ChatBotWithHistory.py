from langchain_openai import ChatOpenAI
import dotenv
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

response = model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)

aimessage = AIMessage(content="Your name is Bob")
