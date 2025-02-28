from langchain_openai import ChatOpenAI
import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
