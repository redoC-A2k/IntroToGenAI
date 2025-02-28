from datetime import datetime
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

def check_weather(location: str, at_time: datetime|None = None) -> str:
    '''Return the weather forecast for the specified location '''
    return f"The weather in {location} is always sunny"

tools = [check_weather]
model = ChatOpenAI(model = "gpt-4o-mini")
graph = create_react_agent(model, tools)
inputs = {"messages": [("user", "What is the weather in sf ?")]}
for s in graph.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        print(message.pretty_print())
# pprint(graph.invoke(inputs))