from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage

from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
    END
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)
from langgraph.checkpoint.memory import InMemorySaver
from tools import search_jira_issues


load_dotenv()


# -------------------------
# 1. Model
# -------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# -------------------------
# 2. Tools
# -------------------------

tools = [
    search_jira_issues
]


model_with_tools = model.bind_tools(
    tools
)


# -------------------------
# 3. LLM node
# -------------------------

def call_model(state: MessagesState):

    response = model_with_tools.invoke(
        [
            SystemMessage(
                content=(
                    "You are an AI project management assistant. "
                    "You have access to live Jira data from the "
                    "Agentic_AI_Workspace project. "
                    "Use the Jira tool whenever the user asks "
                    "about project tasks, issues, statuses, "
                    "priorities, or work items."
                )
            )
        ]
        + state["messages"]
    )

    return {
        "messages": [response]
    }


# -------------------------
# 4. Tool node
# -------------------------

tool_node = ToolNode(
    tools
)


# -------------------------
# 5. Build graph
# -------------------------

builder = StateGraph(
    MessagesState
)


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


# agent = builder.compile()
checkpointer = InMemorySaver()

agent = builder.compile(
    checkpointer=checkpointer
)

# -------------------------
# 6. Run agent
# -------------------------

# result = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": (
#                     "What is the capital of India"
#                     "?"
#                 )
#             }
#         ]
#     }
# )
# -------------------------
# Memory test
# -------------------------

config = {
    "configurable": {
        "thread_id": "ritam-test-1"
    }
}


# First interaction
result1 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "My name is Ritam."
            }
        ]
    },
    config
)

print("\nFIRST RESPONSE:\n")
print(result1["messages"][-1].content)


# Second interaction
result2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is my name?"
            }
        ]
    },
    config
)

print("\nSECOND RESPONSE:\n")
print(result2["messages"][-1].content)


# -------------------------
# 7. Final answer
# -------------------------

# print("\nFINAL ANSWER:\n")

# print(
#     result["messages"][-1].content
# )