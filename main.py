from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field 
from schemas import AgentResponse

class Source(BaseModel):
    """Schema for the source of the information"""
    url: str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the response of the agent"""
    answer: str = Field(description="The agent's answer to the user's question")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to answer the question")

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o")


agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
                }
            ]
        }
    )
    # Access structured response from the agent
    #structured = result.get("structured_response", None)
    #print(structured if structured is not None else result)
    print(result)


if __name__ == "__main__":
    main()
