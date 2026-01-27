from typing import Optional

from langchain.tools import tool, BaseTool
from langchain_community.utilities import SQLDatabase


def database_tool_factory(
        db_client: SQLDatabase, 
        tool_name: str, 
        tool_description: str, 
        static_query: Optional[str] = None,
    ) -> BaseTool:

    def runner(query: str) -> str:
        try:
            return db_client.run(query)
        except Exception as e:
            return f"Error querying database: {e}"

    if static_query:
        @tool(name_or_callable=tool_name, description=tool_description)
        def db_tool() -> str:
            return runner(static_query)
    
    else:
        @tool(name_or_callable=tool_name, description=tool_description)
        def db_tool(query: str) -> str:
            return runner(query)
            
    return db_tool
