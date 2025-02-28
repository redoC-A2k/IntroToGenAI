import uuid
import asyncio
from typing import Optional, List, Dict, Any
import json
import sys
from multi_agent_orchestrator.classifiers import ClassifierResult
from multi_agent_orchestrator.agents import OpenAIAgent, OpenAIAgentOptions, AgentCallbacks, AgentResponse
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.types import ConversationMessage

agent = OpenAIAgent(OpenAIAgentOptions(
    name='Open AI Assistant',
    description='A versatile AI assistant',
    # api_key='' # load from env
))

orchestrator = MultiAgentOrchestrator()
# orchestrator.add_agent(agent)
async def run():
    response =classifier_result = ClassifierResult(selected_agent=agent, confidence=1.0)
    response = await orchestrator.agent_process_request(
        "What is the capital of France?",
        "user123",
        "session456",
        classifier_result
    )
    print(response.output.content[0]['text'])

asyncio.run(run())