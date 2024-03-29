#!/usr/bin/env python
"""Example LangChain server exposes a conversational retrieval agent.

Relevant LangChain documentation:

* Creating a custom agent: https://python.langchain.com/docs/modules/agents/how_to/custom_agent
* Streaming with agents: https://python.langchain.com/docs/modules/agents/how_to/streaming#custom-streaming-with-events
* General streaming documentation: https://python.langchain.com/docs/expression_language/streaming

**ATTENTION**
1. To support streaming individual tokens you will need to use the astream events
   endpoint rather than the streaming endpoint.
2. This example does not truncate message history, so it will crash if you
   send too many messages (exceed token length).
3. The playground at the moment does not render agent output well! If you want to
   use the playground you need to customize it's output server side using astream
   events by wrapping it within another runnable.
4. See the client notebook it has an example of how to use stream_events client side!
"""
from typing import Any

from fastapi import FastAPI
from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel
from langchain.tools.render import format_tool_to_openai_function
from langchain.vectorstores import FAISS

from langserve import add_routes
from langchain.document_loaders import DirectoryLoader, TextLoader

csv_loader = DirectoryLoader('C:/Users/Huynhlong/AI_lord/data', glob="**/*.xlsx")
csv_docs = csv_loader.load()

# DirectoryLoader cho file .txt
text_loader = DirectoryLoader('C:/Users/Huynhlong/AI_lord/data', glob="**/*.txt")
text_docs = text_loader.load()

# Kết hợp dữ liệu từ cả hai loại loader
docs = text_docs + csv_docs

embeddings = OpenAIEmbeddings()

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()


@tool
def get_answer(query: str) -> list:
    """use this tool if user ask something revelant about Nguyen Tat Thanh"""
    return retriever.get_relevant_documents(query)


tools = [get_answer]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "answer only with Vietnamse"
         "You are a college name Nguyen Tat Thanh"
         " your mission is to help user know more about this school"
         "if user appreciate you, you should apreciate them back"
         "you are here to help them know more about the school, not for a chit chat"
         ),
        # Please note that the ordering of the user input vs.
        # the agent_scratchpad is important.
        # The agent_scratchpad is a working space for the agent to think,
        # invoke tools, see tools outputs in order to respond to the given
        # user input. It has to come AFTER the user input.
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# We need to set streaming=True on the LLM to support streaming individual tokens.
# Tokens will be available when using the stream_log / stream events endpoints,
# but not when using the stream endpoint since the stream implementation for agent
# streams action observation pairs not individual tokens.
# See the client notebook that shows how to use the stream events endpoint.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_functions(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using LangChain's Runnable interfaces",
)


# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.
class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: Any


# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
# /stream_events
add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    ),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)