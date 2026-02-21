import os
import sys
sys.path.append(os.getcwd())

import asyncio
import datetime
import json
import random

from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_agent, AgentState

from src.langchain_syntax.agents.mcp.utils import sanitize_mcp_tool
from src.langchain_syntax.agents.tools.web_search import web_search
from src.langchain_syntax.agents.tools.database import database_tool_factory
from src.langchain_syntax.llm.factory import get_mistral


async def init_travel_agent():
    travel_mcp_client = MultiServerMCPClient(
        {
            "travel_server": {
                    "transport": "streamable_http",
                    "url": "https://mcp.kiwi.com"
                }
        }
    )

    travel_tools = await travel_mcp_client.get_tools()
    travel_tools = [sanitize_mcp_tool(tool) for tool in travel_tools]  # sanitize_mcp_tool - костыль для Mistralai
    
    travel_agent = create_agent(
        model=get_mistral(),
        tools=travel_tools,
        system_prompt="""
You are a travel agent. Search for flights to the desired destination wedding location.
You are not allowed to ask any more follow up questions, you must find the best flight options based on the following criteria:
- Price (lowest, economy class)
- Duration (shortest)
- Date (time of year which you believe is best for a wedding at this location)
To make things easy, only look for one ticket, one way.
You may need to make multiple searches to iteratively find the best options.
You will be given no extra information, only the origin and destination. It is your job to think critically about the best options.
Once you have found the best options, let the user know your shortlist of options."""
    )

    return travel_agent


def init_dj_agent():
    dj_client = SQLDatabase.from_uri("sqlite:///src/data/Chinook.db")

    dj_playlist_tool = database_tool_factory(
        db_client=dj_client,
        tool_name="query_playlist_db",
        tool_description="Query the database for playlist information",
    )

    dj_check_tables_tool = database_tool_factory(
        db_client=dj_client,
        tool_name="check_tables",
        tool_description="To check table names before generating query",
        static_query="SELECT name FROM sqlite_master WHERE type='table';",
    )

    dj_agent = create_agent(
        model=get_mistral(),
        tools=[dj_playlist_tool, dj_check_tables_tool],
        system_prompt="""
You are a playlist specialist. Query the sql database and curate the perfect playlist for a wedding given a genre.
Once you have your playlist, calculate the total duration and cost of the playlist, each song has an associated price.
If you run into errors when querying the database, try to fix them by making changes to the query.
Do not come back empty handed, keep trying to query the db until you find a list of songs.
You may need to make multiple queries to iteratively find the best options."""
    )

    return dj_agent


def init_venue_agent():
    venue_agent = create_agent(
        model=get_mistral(),
        tools=[web_search],
        system_prompt="""
You are a venue specialist. Search for venues in the desired location, and with the desired capacity.
You are not allowed to ask any more follow up questions, you must find the best venue options based on the following criteria:
- Price (lowest)
- Capacity (exact match)
- Reviews (highest)
You may need to make multiple searches to iteratively find the best options."""
    )
    
    return venue_agent


async def init_coordinator_agent():
    """Мульти-агент организатор свадеб"""

    # Организатор перелёта
    travel_agent = await init_travel_agent()

    # DJ
    dj_agent = init_dj_agent()

    # Организатор площадки
    venue_agent = init_venue_agent()

    # Runtime state агентов
    class WeddingState(AgentState):
        origin: str
        destination: str
        guest_count: str
        genre: str
        wedding_date: str

    @tool
    def update_state(
        origin: str, 
        destination: str, 
        guest_count: str, 
        genre: str, 
        wedding_date: str, 
        runtime: ToolRuntime,
    ) -> str:
        """Update the state when you know all of the values: origin, destination, guest_count, genre, wedding_date"""
        return Command(update={
            "origin": origin, 
            "destination": destination, 
            "guest_count": guest_count, 
            "genre": genre, 
            "wedding_date": wedding_date,
            "messages": [ToolMessage("Successfully updated state", tool_call_id=runtime.tool_call_id)]}
            )
    
    # Обертка агентов в виде инструмента
    @tool
    async def search_flights(runtime: ToolRuntime) -> str:
        """Travel agent searches for flights to the desired destination wedding location."""
        origin = runtime.state["origin"]
        destination = runtime.state["destination"]
        wedding_date = runtime.state["wedding_date"]
        response = await travel_agent.ainvoke(
            {"messages": [HumanMessage(content=f"Find flights from {origin} to {destination} in {wedding_date}")]}
        )
        return response["messages"][-1].content

    @tool
    def search_venues(runtime: ToolRuntime) -> str:
        """Venue agent chooses the best venue for the given location and capacity."""
        destination = runtime.state["destination"]
        capacity = runtime.state["guest_count"]
        query = f"Find wedding venues in {destination} for {capacity} guests"
        response = venue_agent.invoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    @tool
    def suggest_playlist(runtime: ToolRuntime) -> str:
        """Playlist agent curates the perfect playlist for the given genre."""
        genre = runtime.state["genre"]
        query = f"Find {genre} tracks for wedding playlist"
        response = dj_agent.invoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    # Агент-планировщик
    coordinator = create_agent(
        model=get_mistral(),
        tools=[search_flights, search_venues, suggest_playlist, update_state],
        state_schema=WeddingState,
        system_prompt="""
You are a wedding coordinator. Delegate tasks to your specialists for flights, venues and playlists.
First find all the information you need to update the state. Once that is done you can delegate the tasks.
Once you have received their answers, coordinate the perfect wedding for me."""
    )

    return coordinator


async def main():
    coordinator = await init_coordinator_agent()

    # Дата свадьбы
    N = random.randint(3, 6)
    wedding_date = (datetime.datetime.now() + datetime.timedelta(weeks=4*N)).date().isoformat()  # Свадьба через N месяцев

    # Место
    origin, destination = "London", "Paris"

    # Количество гостей
    guest_count = str(random.randint(30, 100))

    # Тест
    query = f"I'm from {origin} and I'd like a wedding in {destination} for {guest_count} guests, jazz-genre. The wedding date is {wedding_date}"
    response = await coordinator.ainvoke({"messages": [HumanMessage(content=query)]})    

    assert response["origin"] == origin
    assert response["destination"] == destination
    assert response["wedding_date"] == wedding_date
    assert response["guest_count"] == guest_count

    response_dict = {
        "messages": [
            {
                str(type(m)): m.model_dump()
            } for m in response["messages"]
        ]
    }

    with open(os.path.join(os.path.dirname(__file__), "multi_agent_output.json"), "w", encoding="utf-8") as f:
        json.dump(response_dict, f, indent=4, ensure_ascii=False)




if __name__ == "__main__":
    asyncio.run(main())
