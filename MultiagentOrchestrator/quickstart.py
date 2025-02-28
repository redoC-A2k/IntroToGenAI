import uuid
import asyncio
from typing import Optional, List, Dict, Any
import json
import sys
from openai import OpenAI
from multi_agent_orchestrator.agents import OpenAIAgent, OpenAIAgentOptions, AgentCallbacks, AgentResponse
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.types import ConversationMessage
from multi_agent_orchestrator.classifiers import OpenAIClassifier, OpenAIClassifierOptions

# Creating orchestrator
openai_classifier = OpenAIClassifier(OpenAIClassifierOptions(
    api_key='Your api key',
))

orchestrator = MultiAgentOrchestrator(options=OrchestratorConfig(
    LOG_AGENT_CHAT=True,
    LOG_CLASSIFIER_CHAT=True,
    LOG_CLASSIFIER_RAW_OUTPUT=True,
    LOG_CLASSIFIER_OUTPUT=True,
    LOG_EXECUTION_TIMES=True,
    MAX_RETRIES=3,
    USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
    MAX_MESSAGE_PAIRS_PER_AGENT=10
), classifier=openai_classifier)

# Adding agents
class OpenAILLMAgentCallbacks(AgentCallbacks):
    def on_llm_new_token(self, token: str) -> None:
        print(token, end='', flush=True)

tech_agent = OpenAIAgent(OpenAIAgentOptions(
    name = "Tech Agent",
    api_key='Your api key',
    streaming=True,
    description="Specializes in technology areas including software development, hardware, AI,cybersecurity, blockchain, cloud computing, emerging tech innovations, and pricing/costs related to technology products and services.",
    callbacks=OpenAILLMAgentCallbacks()
))

orchestrator.add_agent(tech_agent)

health_agent =  OpenAIAgent(OpenAIAgentOptions(
    name="Health Agent",
    # streaming=True,
    api_key='Your api key',
    description="Focuses on health and medical topics such as general wellness, nutrition, diseases, treatments, mental health, fitness, healthcare systems, and medical terminology or concepts.",
    callbacks=OpenAILLMAgentCallbacks()
))

orchestrator.add_agent(health_agent)

# Sending query
async def handle_request(_orchestrator: MultiAgentOrchestrator, _user_input:str, _user_id:str, _session_id: str):
    response: AgentResponse = await _orchestrator.route_request(_user_input, _user_id, _session_id)
    print("\nMetadata: ", )
    print(f"Selected Agent: {response.metadata.agent_name}")
    print("\nResponse: ")    
    print(response.output)
    if response.streaming:
        print('Response:', response.output.content[0]['text'])
    else:
        print('Response:', response.output.content[0]['text'])

if __name__=="__main__":
    USER_ID = "user123"
    SESSION_ID = str(uuid.uuid4())
    print("Welcome to interactive MultiAgent System. Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye ! ")
            sys.exit(0)
        
        asyncio.run(handle_request(orchestrator, user_input, USER_ID, SESSION_ID))