from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition


# ==================================================
# Shared model factory
# ==================================================

def get_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )


# ==================================================
# Jira Agent
# ==================================================

def build_jira_agent(jira_tools, jira_cloud_id):

    model = get_model()
    model_with_tools = model.bind_tools(jira_tools)

    async def call_model(state: MessagesState):

        response = await model_with_tools.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are the Jira specialist agent.\n\n"

                        "Your only responsibility is to retrieve accurate "
                        "current information from Jira.\n\n"

                        "The Jira project is Agentic_AI_Workspace.\n"
                        "The Jira project key is AAW.\n"
                        f"The Jira cloudId is {jira_cloud_id}.\n\n"

                        "For searches concerning this project, ALWAYS use "
                        "project = AAW in the JQL.\n\n"

                        "When using searchJiraIssuesUsingJql, provide the "
                        f"cloudId {jira_cloud_id}.\n\n"

                        "Do not invent Jira information.\n"
                        "Use Jira MCP tools whenever Jira data is required.\n"
                        "Return factual findings clearly so the supervisor "
                        "agent can use them."
                    )
                )
            ] + state["messages"]
        )

        return {
            "messages": [response]
        }

    tool_node = ToolNode(jira_tools)

    builder = StateGraph(MessagesState)

    builder.add_node("llm", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "llm")

    builder.add_conditional_edges(
        "llm",
        tools_condition
    )

    builder.add_edge(
        "tools",
        "llm"
    )

    return builder.compile()


# ==================================================
# Notion Agent
# ==================================================

def build_notion_agent(notion_tools):

    model = get_model()
    model_with_tools = model.bind_tools(notion_tools)

    async def call_model(state: MessagesState):

        response = await model_with_tools.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are the Notion specialist agent.\n\n"

                        "Your only responsibility is to retrieve accurate "
                        "project information from Notion.\n\n"

                        "The primary project is Agentic AI Workspace.\n\n"

                        "Use notion-search to find relevant pages and "
                        "notion-fetch to retrieve page contents when needed.\n\n"

                        "Focus on project documentation, plans, architecture, "
                        "tasks, issues, and decisions related to Agentic AI Workspace.\n\n"

                        "Do not invent Notion information.\n"
                        "Return factual findings clearly so the supervisor "
                        "agent can use them."
                    )
                )
            ] + state["messages"]
        )

        return {
            "messages": [response]
        }

    tool_node = ToolNode(notion_tools)

    builder = StateGraph(MessagesState)

    builder.add_node("llm", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "llm")

    builder.add_conditional_edges(
        "llm",
        tools_condition
    )

    builder.add_edge(
        "tools",
        "llm"
    )

    return builder.compile()