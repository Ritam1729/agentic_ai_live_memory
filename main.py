import os
import base64
import asyncio

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, ToolMessage

from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import PostgresStore

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import tool

from tools import (
    save_memory,
    search_memory,
)
from evidence import create_evidence

from agents import (
    build_jira_agent,
    build_notion_agent,
)

class AgentState(MessagesState):
    evidence: list[dict]
    reconciliation: dict

if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )
# ==================================================
# 1. Environment
# ==================================================

load_dotenv()

POSTGRES_URI = os.getenv("POSTGRES_URI")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_CLOUD_ID = os.getenv("JIRA_CLOUD_ID")

if not JIRA_CLOUD_ID:
    raise ValueError("JIRA_CLOUD_ID is not set in .env")

if not POSTGRES_URI:
    raise ValueError("POSTGRES_URI is not set in .env")

if not JIRA_EMAIL:
    raise ValueError("JIRA_EMAIL is not set in .env")

if not JIRA_API_TOKEN:
    raise ValueError("JIRA_API_TOKEN is not set in .env")


# ==================================================
# 2. Jira MCP authentication
# ==================================================

credentials = f"{JIRA_EMAIL}:{JIRA_API_TOKEN}"

encoded_credentials = base64.b64encode(
    credentials.encode()
).decode()

jira_auth_header = {
    "Authorization": f"Basic {encoded_credentials}"
}


# ==================================================
# 3. Model
# ==================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)


# ==================================================
# 4. Main application
# ==================================================

async def main():

    # --------------------------------------------------
    # 4.1 Connect to Atlassian Rovo MCP
    # --------------------------------------------------

    client = MultiServerMCPClient(
        {
            "atlassian": {
                "transport": "http",
                "url": "https://mcp.atlassian.com/v1/mcp",
                "headers": jira_auth_header,
            },

            "notion": {
                "transport": "stdio",
                "command": "npx",
                "args": [
                    "-y",
                    "mcp-remote",
                    "https://mcp.notion.com/mcp",
                ],
            },
        }
    )

    # Get tools exposed by the MCP servers
    mcp_tools = await client.get_tools()

    jira_tools = [
        tool for tool in mcp_tools
        if (
            tool.name.startswith("getJira")
            or tool.name.startswith("searchJira")
            or tool.name.startswith("lookupJira")
        )
    ]

    notion_tools = [
        tool for tool in mcp_tools
        if tool.name.startswith("notion-")
    ]

    jira_agent = build_jira_agent(
    jira_tools,
    JIRA_CLOUD_ID
    )
    notion_agent = build_notion_agent(notion_tools)

    @tool
    async def ask_jira_agent(question: str) -> str:
        """
        Ask the Jira specialist to research current Jira information.
        """

        result = await jira_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        return result["messages"][-1].content

    @tool
    async def ask_notion_agent(question: str) -> str:
        """
        Ask the Notion specialist to research project documentation.
        """

        result = await notion_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        return result["messages"][-1].content

    print("\n" + "=" * 60)
    print("MCP TOOLS DISCOVERED")
    print("=" * 60)

    for mcp_tool in mcp_tools:
        print(f"- {mcp_tool.name}")

    # --------------------------------------------------
    # 4.2 Supervisor tools
    # --------------------------------------------------

    supervisor_tools = [
        save_memory,
        search_memory,
        ask_jira_agent,
        ask_notion_agent,
    ]

    print(
        f"SUPERVISOR TOOLS AVAILABLE: {len(supervisor_tools)}"
    )
    # --------------------------------------------------
    # 4.4 Bind supervisor tools to Gemini
    # --------------------------------------------------

    model_with_tools = model.bind_tools(
        supervisor_tools
    )


    # ==================================================
    # 5. LLM node
    # ==================================================

    async def call_model(state: MessagesState):

        response = await model_with_tools.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are the supervisor agent for an AI "
                        "project-management workspace.\n\n"

                        "Your job is to understand the user's request, "
                        "delegate research to the appropriate specialist "
                        "agents, compare their findings, and produce the "
                        "final answer.\n\n"

                        "SPECIALIST AGENTS:\n"
                        "- ask_jira_agent: retrieves current Jira "
                        "information for project AAW.\n"
                        "- ask_notion_agent: retrieves project "
                        "documentation and plans from Notion.\n"
                        "- search_memory: retrieves relevant long-term "
                        "memory.\n"
                        "- save_memory: stores durable information when "
                        "appropriate.\n\n"

                        "DELEGATION RULES:\n"
                        "- Use ask_jira_agent for current Jira tasks, "
                        "issues, statuses, priorities, and operational "
                        "project state.\n"
                        "- Use ask_notion_agent for project documentation, "
                        "plans, architecture, decisions, and documented "
                        "status.\n"
                        "- Use search_memory when previous conversations "
                        "may be relevant.\n"
                        "- When the user asks for a comparison between "
                        "Jira and Notion, use BOTH specialist agents.\n\n"

                        "RECONCILIATION:\n"
                        "- Compare the findings returned by the specialists.\n"
                        "- Identify agreements, differences, and conflicts.\n"
                        "- Do not silently hide conflicting information.\n"
                        "- Prefer Jira for current operational task status.\n"
                        "- Prefer Notion for project documentation and plans.\n\n"

                        "EVIDENCE:\n"
                        "- Base factual claims on specialist results or "
                        "retrieved memory.\n"
                        "- Do not invent facts or sources.\n"
                        "- Clearly identify which source supports important "
                        "claims.\n\n"

                        "Always delegate to the appropriate specialist when "
                        "external information is needed. Do not guess."
                    )
                )
            ] + state["messages"]
        )

        return {
            "messages": [response]
        }

    tool_node = ToolNode(
        supervisor_tools
    )

    def collect_evidence(state: AgentState):

        evidence_items = []

        for message in state["messages"]:

            if isinstance(message, ToolMessage):

                tool_name = message.name or "unknown_tool"

                content = message.content

                evidence = create_evidence(
                    source="MCP",
                    source_id=tool_name,
                    content=str(content),
                    metadata={
                        "tool_name": tool_name,
                        "tool_call_id": message.tool_call_id,
                    },
                )

                evidence_items.append(evidence)

        return {
            "evidence": evidence_items
        }

    def reconcile_evidence(state: AgentState):

        evidence = state.get("evidence", [])

        if not evidence:
            return {
                "reconciliation": {
                    "agreements": [],
                    "conflicts": [],
                    "unresolved": []
                }
            }

        # For now, let the LLM perform the semantic comparison
        # using the collected evidence.
        return {
            "reconciliation": {
                "evidence_count": len(evidence),
                "sources": list({
                    item["source"]
                    for item in evidence
                })
            }
        }



    # ==================================================
    # 7. Build graph
    # ==================================================

    builder = StateGraph(MessagesState)


    builder.add_node(
        "llm",
        call_model
    )

    builder.add_node(
        "tools",
        tool_node
    )

    builder.add_edge(
        START,
        "llm"
    )

    builder.add_conditional_edges(
        "llm",
        tools_condition
    )

    builder.add_edge(
        "tools",
        "llm"
    )

    # ==================================================
    # 8. Embedding model
    # ==================================================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # ==================================================
    # 9. PostgreSQL persistence + semantic memory
    # ==================================================

    async with AsyncPostgresSaver.from_conn_string(
        POSTGRES_URI
    ) as checkpointer:

        await checkpointer.setup()


        with PostgresStore.from_conn_string(
            POSTGRES_URI,
            index={
                "dims": 384,
                "embed": embeddings,
                "fields": ["memory"],
            },
        ) as store:

            store.setup()


            # --------------------------------------------------
            # Compile graph
            # --------------------------------------------------

            agent = builder.compile(
                checkpointer=checkpointer,
                store=store,
            )


            # ==================================================
            # 10. Conversation
            # ==================================================

            config = {
                "configurable": {
                    "thread_id": "mcp-test-1"
                }
            }


            result = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (  
                                        "Compare the current in-progress tasks in Jira with the "
                                        "tasks documented in my Agentic AI Workspace Notion page. "
                                        "Identify agreements and differences."
                                        ),
                        }
                    ],
                    "evidence": [],
                },
                config,
            )


            # ==================================================
            # 11. Final answer
            # ==================================================

            print("\n" + "=" * 60)
            print("FINAL ANSWER")
            print("=" * 60)

            print(
                result["messages"][-1].content
            )


# ==================================================
# 12. Run
# ==================================================

if __name__ == "__main__":
    asyncio.run(main())