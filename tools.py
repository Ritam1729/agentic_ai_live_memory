import os
import uuid
import requests

from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

from langchain.tools import tool, ToolRuntime


load_dotenv()


# --------------------------------------------------
# Jira
# --------------------------------------------------

JIRA_URL = os.getenv("JIRA_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

PROJECT_KEY = "AAW"


@tool
def search_jira_issues(
    status: str | None = None,
    priority: str | None = None
) -> str:
    """
    Search Jira issues in the Agentic AI Workspace project.
    """

    jql_parts = [
        f"project = {PROJECT_KEY}"
    ]

    if status:
        jql_parts.append(
            f'status = "{status}"'
        )

    if priority:
        jql_parts.append(
            f'priority = "{priority}"'
        )

    jql = " AND ".join(jql_parts)
    jql += " ORDER BY created DESC"

    url = f"{JIRA_URL}/rest/api/3/search/jql"

    params = {
        "jql": jql,
        "maxResults": 20,
        "fields": "summary,status,priority,assignee"
    }

    response = requests.get(
        url,
        params=params,
        auth=HTTPBasicAuth(
            JIRA_EMAIL,
            JIRA_API_TOKEN
        ),
        headers={
            "Accept": "application/json"
        }
    )

    if not response.ok:
        return (
            f"Jira API request failed. "
            f"Status code: {response.status_code}. "
            f"Response: {response.text}"
        )

    data = response.json()

    issues = data.get("issues", [])

    if not issues:
        return "No matching Jira issues were found."

    results = []

    for issue in issues:

        fields = issue["fields"]

        summary = fields.get(
            "summary",
            "No summary"
        )

        status_name = fields.get(
            "status",
            {}
        ).get(
            "name",
            "Unknown"
        )

        priority_data = fields.get("priority")

        priority_name = (
            priority_data.get("name", "Unknown")
            if priority_data
            else "None"
        )

        assignee_data = fields.get("assignee")

        assignee_name = (
            assignee_data.get("displayName", "Unknown")
            if assignee_data
            else "Unassigned"
        )

        results.append(
            f"{issue['key']} | "
            f"{summary} | "
            f"Status: {status_name} | "
            f"Priority: {priority_name} | "
            f"Assignee: {assignee_name}"
        )

    return "\n".join(results)


# --------------------------------------------------
# Long-term memory
# --------------------------------------------------

@tool
def save_memory(
    memory: str,
    runtime: ToolRuntime
) -> str:
    """
    Save an important long-term memory about the user.

    Use this for durable user preferences, important project facts,
    decisions, or information the user explicitly asks to remember.
    Do not save temporary or ordinary conversational information.
    """

    user_id = "ritam"

    memory_id = str(uuid.uuid4())

    runtime.store.put(
        (user_id, "memories"),
        memory_id,
        {
            "memory": memory
        }
    )

    print("\n[SAVE_MEMORY]")
    print("Memory:", memory)

    return "Memory saved successfully."

@tool
def search_memory(
    query: str,
    runtime: ToolRuntime
) -> str:
    """
    Search the user's long-term memories.
    """

    user_id = "ritam"

    results = runtime.store.search(
        (user_id, "memories"),
        query=query,
        limit=5
    )

    print("\n[SEARCH_MEMORY]")
    print("Query:", query)
    print("Results:", len(results))

    if not results:
        return "No relevant memories were found."

    memories = []

    for item in results:

        memory = item.value.get("memory")

        if memory:
            memories.append(memory)

    return "\n".join(
        f"- {memory}"
        for memory in memories
    )