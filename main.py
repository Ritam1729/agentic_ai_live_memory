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
from evidence import create_evidence
import asyncio

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
                        "You are an AI project management assistant with access to "
                        "live Jira,and Notion data, persistent conversation state, "
                        "and persistent long-term memory.\n\n"

                        "RESPONSIBILITIES:\n"
                        "1. Answer accurately, clearly, and concisely.\n"
                        "2. Use live external data when the user's question requires "
                        "current information.\n"
                        "3. Use long-term memory when information from previous "
                        "conversations may be relevant.\n"
                        "4. Save information to long-term memory when the user explicitly "
                        "asks you to remember it, or when it is a durable preference, "
                        "project fact, or important decision.\n\n"

                        "MEMORY RULES:\n"
                        "- If the user says 'remember this', 'save this', "
                        "'keep this in mind', or similar, ALWAYS call save_memory "
                        "before responding.\n"
                        "- Do not save ordinary, temporary, or irrelevant conversation.\n"
                        "- Use search_memory when the answer may depend on information "
                        "from previous conversations.\n"
                        "- Never claim to remember something unless it was actually "
                        "retrieved from memory or is present in the current conversation.\n\n"

                        "JIRA RULES:\n"
                        "- Use Jira MCP tools for current Jira tasks, issues, statuses, "
                        "priorities, assignees, and other live Jira information.\n"
                        "- The primary Jira project is Agentic_AI_Workspace with project "
                        "key AAW.\n"
                        "- For questions about this project, ALL Jira searches must "
                        "include project = AAW in the JQL.\n"
                        "- Never search unrelated Jira projects unless the user explicitly "
                        "asks for another project.\n"
                        "- For current in-progress issues use:\n"
                        "  project = AAW AND status = \"In Progress\"\n"
                        "- Prefer live Jira data over memory for current Jira state.\n"
                        "- Never invent Jira issues or Jira information.\n\n"

                        "NOTION RULES:\n"
                        "- Use Notion MCP tools for project documentation, plans, "
                        "architecture notes, specifications, decisions, and project notes.\n"
                        "- Use notion-search to find relevant pages.\n"
                        "- Use notion-fetch to retrieve the contents of relevant pages.\n"
                        "- Do not invent Notion content.\n\n"

                        "SOURCE PRIORITY:\n"
                        "- Jira is the preferred source for current task status and "
                        "operational project state.\n"
                        "- Notion is the preferred source for project documentation, "
                        "plans, architecture, and documented decisions.\n"
                        "- If sources disagree, do not silently choose one. Explicitly "
                        "identify the conflict and show the relevant evidence.\n\n"

                        "CROSS-SOURCE REASONING:\n"
                        "- When the user asks for information across Jira, and Notion"
                        " query the relevant sources rather than answering from "
                        "only one source.\n"
                        "- Compare the retrieved information across sources.\n"
                        "- Identify agreements, differences, and missing information.\n"
                        "- Do not assume that similar wording means the sources agree.\n"
                        "- When information conflicts, report the conflict explicitly.\n\n"

                        "EVIDENCE RULES:\n"
                        "- Treat external tool results as evidence.\n"
                        "- Base factual claims about Jira,and Notion on retrieved "
                        "evidence.\n"
                        "- Jira evidence should be cited as [Jira: tool_name].\n"
                        "- Notion evidence should be cited as [Notion: tool_name].\n"
                        "- Do not invent citations or sources.\n"
                        "- If the available evidence does not support a claim, say so.\n"
                        "- When sources disagree, cite the relevant sources separately.\n\n"

                        "TOOL USAGE:\n"
                        "- Use the minimum number of tools necessary, but use ALL relevant "
                        "sources when the user explicitly asks for a cross-source comparison.\n"
                        "- After receiving a tool result, determine whether another tool "
                        "call is necessary before answering.\n"
                        "- Do not claim that an external action succeeded unless the "
                        "corresponding tool returned a successful result."
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
    builder.add_node(
        "reconciliation",
        reconcile_evidence
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
        "reconciliation"
    )

    builder.add_edge(
        "reconciliation",
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
                                                "Find all currently in-progress tasks and issues related to "
                                                "my Agentic AI Workspace. Check Jira,and the relevant Notion page"
                                                ". Combine the information, identify which "
                                                "items appear across multiple sources, and clearly identify "
                                                "any differences or conflicts."
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