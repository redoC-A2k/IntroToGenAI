from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model = "gpt-4o-mini")
workflow = StateGraph(state_schema=MessagesState)

# def call_model(state: MessagesState):
#     return {"messages": [model.invoke(state["messages"])]}

# workflow.add_edge(START, "model")
# workflow.add_node("model", call_model)

# memory = MemorySaver()
# graph = workflow.compile(checkpointer=memory)
# config = {"configurable": {"thread_id": "abc123"}}

# query = "Hi ! I am afshan"
# input_messages = [HumanMessage(query)]
# output = graph.invoke({"messages": input_messages}, config)
# print(output["messages"][-1].pretty_print())

# query = "What is my name ?"
# input_messages = [HumanMessage(query)]
# output = graph.invoke({"messages": input_messages}, config)
# print(output["messages"][-1].pretty_print())

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# prompt_template = ChatPromptTemplate.from_messages([
#     ("system","You talk like a pirate. Answer all question to the best of your ability."),
#     MessagesPlaceholder(variable_name="messages"),
# ])

# def call_model(state: MessagesState):
#     print("-----------")
#     print(state)
#     print("-----------")
#     prompt = prompt_template.invoke(state)
#     response = model.invoke(prompt)
#     return {"messages": [response]}

# workflow.add_edge(START, "model")
# workflow.add_node("model", call_model)

# memory = MemorySaver()
# graph = workflow.compile(checkpointer=memory)

# config = {"configurable": {"thread_id": "abc345"}} 
# query = "Hi ! I am Thanos"
# input_messages = [HumanMessage(query)]
# output = graph.invoke({"messages": input_messages}, config)
# print(output["messages"][-1].pretty_print())

# query = "What is my name ?"
# input_messages = [HumanMessage(query)]
# output = graph.invoke({"messages": input_messages}, config)
# print(output["messages"][-1].pretty_print())

from typing import Sequence
from langchain_core.messages import BaseMessage, trim_messages
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

prompt_template = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
    MessagesPlaceholder(variable_name="messages"),
])
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

graph_builder = StateGraph(state_schema=State)
trimmer = trim_messages(
    max_tokens = 65,
    strategy = "last",
    token_counter = model,
    include_system = True,
    allow_partial = False,
    start_on = "human"

)
def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({"messages": trimmed_messages, "language": state["language"]})
    # prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}

graph_builder.add_edge(START, "model")
graph_builder.add_node("model", call_model)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}
# query = "Hi ! I am iron man"
# language = "English"

# input_messages = [HumanMessage(query)]
# output = graph.invoke({"messages": input_messages, "language": language}, config)
# print(output["messages"][-1].pretty_print())

query = "What is my name ?"
language = "English"

input_messages = [HumanMessage(query)]
output = graph.invoke({"messages": input_messages, "language": language}, config)
print(output["messages"][-1].pretty_print())