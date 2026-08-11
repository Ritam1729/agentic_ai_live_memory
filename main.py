from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from tools import get_project_status


load_dotenv()


# -------------------------
# 1. Model
# -------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# -------------------------
# 2. Give the model tools
# -------------------------

tools = [get_project_status]

model_with_tools = model.bind_tools(tools)


# -------------------------
# 3. LLM node
# -------------------------

def call_model(state: MessagesState):

    response = model_with_tools.invoke(
        [
            SystemMessage(
                content=(
                    "You are a project management assistant. "
                    "Use the available tools whenever you need "
                    "project information."
                )
            )
        ] + state["messages"]
    )

    return {
        "messages": [response]
    }


# -------------------------
# 4. Tool node
# -------------------------

tool_node = ToolNode(tools)


# -------------------------
# 5. Build graph
# -------------------------

builder = StateGraph(MessagesState)

builder.add_node("llm", call_model)
builder.add_node("tools", tool_node)

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


agent = builder.compile()


# -------------------------
# 6. Run agent
# -------------------------
print(agent.get_graph().draw_mermaid())
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the deadline of Project X?"
            }
        ]
    }
)


# -------------------------
# 7. Print final answer
# -------------------------

print("\nFINAL ANSWER:\n")

print(result["messages"][-1].content)