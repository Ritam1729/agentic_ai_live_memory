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

from tools import (
    save_memory,
    search_memory,
)

import asyncio
class AgentState(MessagesState):
    evidence: list[dict]

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
    model="gemini-3.6-flash",
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
            }
        }
    )

    # Get tools exposed by the MCP server
    mcp_tools = await client.get_tools()


    print("\n" + "=" * 60)
    print("MCP TOOLS DISCOVERED")
    print("=" * 60)

    for tool in mcp_tools:
        print(f"- {tool.name}")


    # --------------------------------------------------
    # 4.2 Local tools
    # --------------------------------------------------

    local_tools = [
        save_memory,
        search_memory,
    ]


    # --------------------------------------------------
    # 4.3 Combine local + MCP tools
    # --------------------------------------------------

    tools = local_tools + mcp_tools


    print("\n" + "=" * 60)
    print(f"TOTAL TOOLS AVAILABLE: {len(tools)}")
    print("=" * 60)


    # --------------------------------------------------
    # 4.4 Bind tools to Gemini
    # --------------------------------------------------

    model_with_tools = model.bind_tools(tools)


    # ==================================================
    # 5. LLM node
    # ==================================================

    async def call_model(state: MessagesState):

        response = await model_with_tools.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are an AI project management assistant "
                        "with access to live Atlassian data and persistent "
                        "long-term memory.\n\n"

                        "RESPONSIBILITIES:\n"
                        "1. Answer accurately and concisely.\n"
                        "2. Use Atlassian MCP tools whenever the user asks "
                        "about current Jira projects, issues, tasks, "
                        "statuses, priorities, assignees, or other live "
                        "Atlassian information.\n"
                        "3. Use search_memory when information from a "
                        "previous conversation may be relevant.\n"
                        "4. Use save_memory when the user explicitly asks "
                        "you to remember something for future conversations "
                        "or when information is clearly durable and useful.\n\n"

                        "MEMORY RULES:\n"
                        "- If the user says 'remember this', 'save this', "
                        "'keep this in mind', or similar, ALWAYS call "
                        "save_memory before responding.\n"
                        "- Do not save ordinary or temporary conversation.\n"
                        "- Before answering a question that may depend on "
                        "previously remembered information, use search_memory.\n"
                        "- Never claim to remember information unless it was "
                        "actually retrieved from memory or exists in the "
                        "current conversation.\n\n"

                        "JIRA RULES:\n"
                            "- Use Atlassian MCP tools for questions about current Jira "
                            "tasks and project status.\n"
                            "- Prefer live Jira information over memory for current Jira state.\n"
                            "- Do not invent Jira issues, statuses, priorities, or other "
                            "project information.\n\n"

                            "JIRA RULES:\n"
                            "- You have access to the Jira project Agentic_AI_Workspace.\n"
                            "- The Jira project key is AAW.\n"
                            "- For ALL Jira searches, restrict the JQL to project = AAW "
                            "unless the user explicitly asks about another project.\n"
                            "- Never search all Jira projects when answering questions "
                            "about this project.\n"
                            "- For example, for in-progress issues use:\n"
                            "  project = AAW AND status = \"In Progress\"\n"
                            "- For high-priority incomplete issues use:\n"
                            "  project = AAW AND priority = High AND status != Done\n"
                            "- Use searchJiraIssuesUsingJql for Jira searches.\n"
                            "- Do not invent Jira issues or project information.\n\n"

                            "JIRA CONTEXT:\n"
                            f"- Jira site: {os.getenv('JIRA_URL')}\n"
                            f"- Jira cloudId: {JIRA_CLOUD_ID}\n"
                            f"- Jira project key: AAW\n"
                            "EVIDENCE RULES:\n"
                            "- Treat tool results as evidence.\n"
                            "- When answering questions using external data, base factual "
                            "claims on the available evidence.\n"
                            "- Cite important external claims using the evidence source "
                            "and source ID.\n"
                            "- Use citations in this format: [MCP: tool_name]\n"
                            "- Do not invent citations or sources.\n"
                            "- If the available evidence does not support a claim, say so.\n\n"
                    )
                )
            ]
            + state["messages"]
        )

        return {
            "messages": [response]
        }

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


    # ==================================================
    # 6. Tool node
    # ==================================================

    tool_node = ToolNode(tools)


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

    builder.add_node(
        "evidence",
        collect_evidence
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
        "evidence"
    )

    builder.add_edge(
        "evidence",
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
                                "What Jira issues are currently "
                                "in progress?"
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