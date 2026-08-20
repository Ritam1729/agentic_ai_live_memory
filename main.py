from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage

from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

#from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
# from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore

from tools import (
    search_jira_issues,
    save_memory,
    search_memory
)
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()

POSTGRES_URI = os.getenv("POSTGRES_URI")


# ==================================================
# 1. Model
# ==================================================

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ==================================================
# 2. Tools
# ==================================================

tools = [
    search_jira_issues,
    save_memory,
    search_memory
]

model_with_tools = model.bind_tools(tools)


# ==================================================
# 3. LLM node
# ==================================================

def call_model(state: MessagesState):

    response = model_with_tools.invoke(
        [
            SystemMessage(
                content=(
                "You are an AI project management assistant with access to "
                "live Jira data and persistent long-term memory.\n\n"

                "YOUR RESPONSIBILITIES:\n"
                "1. Answer the user's questions accurately and concisely.\n"
                "2. Use live Jira data whenever the question concerns the "
                "current state of projects, issues, tasks, statuses, priorities, "
                "assignees, or other Jira information.\n"
                "3. Use long-term memory when information from previous "
                "conversations may be relevant to the user's request.\n"
                "4. Save information to long-term memory when the user "
                "explicitly asks you to remember something for future "
                "conversations, or when the information is clearly a useful "
                "and durable preference, project fact, or decision."
                "5. Search the saved memory whenever needed.\n\n"

                "MEMORY RULES:\n"
                "- If the user explicitly says 'remember this', 'save this', "
                "'keep this in mind', or similar, ALWAYS call save_memory "
                "before giving your final response.\n"
                "- Do not save ordinary conversation, temporary information, "
                "or information that is unlikely to be useful later.\n"
                "- Before answering a question that may depend on a previous "
                "preference, fact, or decision, use search_memory.\n"
                "- Do not claim to remember something unless it was actually "
                "retrieved from long-term memory or is present in the current "
                "conversation.\n\n"

                "JIRA RULES:\n"
                "- Use search_jira_issues for questions about current Jira "
                "tasks and project status.\n"
                "- Prefer live Jira information over older conversational "
                "information when the user asks about the current state.\n"
                "- Do not invent Jira issues, statuses, priorities, or other "
                "project information.\n\n"

                "TOOL USAGE:\n"
                "- Use the minimum number of tools necessary to answer the "
                "user's question.\n"
                "- When a tool provides the required information, use that "
                "information in your final answer rather than guessing.\n"
                "- After receiving a tool result, decide whether another tool "
                "call is necessary before answering.\n"
                "- Never tell the user that you performed an action unless "
                "the corresponding tool call actually succeeded."
                )
            )
        ] + state["messages"]
    )

    return {
        "messages": [response]
    }


# ==================================================
# 4. Tool node
# ==================================================

tool_node = ToolNode(tools)


# ==================================================
# 5. Build graph
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



# store = InMemoryStore()


# ==================================================
# 6. Embedding model
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# 7. Persistent PostgreSQL checkpoint + memory store
# ==================================================

with PostgresSaver.from_conn_string(
    POSTGRES_URI
) as checkpointer:

    checkpointer.setup()

    with PostgresStore.from_conn_string(
        POSTGRES_URI,
        index={
            "dims": 384,
            "embed": embeddings,
            "fields": ["memory"]
        }
    ) as store:

        store.setup()

        agent = builder.compile(
            checkpointer=checkpointer,
            store=store
        )

        # ==========================================
        # Conversation
        # ==========================================

        config = {
            "configurable": {
                "thread_id": "new-conversation-1"
            }
        }

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "How you should reply to my questions?"
                            
                        )
                    }
                ]
            },
            config
        )

        print("\nFINAL RESPONSE:\n")
        print(result["messages"][-1].content)
    # print(
    # store.search(("user_memory",))
    # )

    # ==============================================
    # Conversation 2
    # ==============================================

    # config_2 = {
    #     "configurable": {
    #         "thread_id": "conversation-2"
    #     }
    # }


    # result2 = agent.invoke(
    #     {
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": (
    #                     "How should you give answer "
    #                     "to me?"
    #                 )
    #             }
    #         ]
    #     },
    #     config_2
    # )


    # print("\nSECOND RESPONSE:\n")
    # print(
    #     result2["messages"][-1].content
    # )