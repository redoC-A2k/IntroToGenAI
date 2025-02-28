from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from pprint import pprint
load_dotenv()

memory = MemorySaver()
model = ChatOpenAI(model = "gpt-4o-mini")
search = TavilySearchResults(max_results=2)
tools = [search]

# Why tools are needed
# print("tools", search.invoke("what is the weather in Indore ?"))
# model_with_tools = model.bind_tools(tools)
# response = model_with_tools.invoke([HumanMessage(content="What is the weather in London")])
# print(f"ContentString", response.content) # No output here
# print(f"ToolCalls:", response.tool_calls) # output -> ToolCalls: [{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in London'}, 'id': 'call_TuZKvTCtDAZtaHUTAuKPRlZE', 'type': 'tool_call'}]
# # From above toolcalls output we can infer that we need to call tool tavilySearchResults

agent_executor = create_react_agent(model, tools)
response = agent_executor.invoke({"messages": [HumanMessage(content="What is the weather in London ?")]})
for message in response["messages"]:
    pprint(message)