import os
import requests

from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from langchain.tools import tool


load_dotenv()


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
    Search issues in the Agentic AI Workspace Jira project.

    Use this tool when the user asks about Jira tasks,
    issues, their status, or priority.

    Args:
        status: Optional Jira status such as
                "To Do", "In Progress", "In Review", or "Done".

        priority: Optional priority such as
                  "High", "Medium", or "Low".
    """

    # Start with the project restriction.
    jql_parts = [
        f"project = {PROJECT_KEY}"
    ]

    # Add status filter if requested.
    if status:
        jql_parts.append(
            f'status = "{status}"'
        )

    # Add priority filter if requested.
    if priority:
        jql_parts.append(
            f'priority = "{priority}"'
        )

    # Construct JQL.
    jql = " AND ".join(jql_parts)

    jql += " ORDER BY created DESC"


    # Jira search endpoint.
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

        priority_data = fields.get(
            "priority"
        )

        if priority_data:
            priority_name = priority_data.get(
                "name",
                "Unknown"
            )
        else:
            priority_name = "None"


        assignee_data = fields.get(
            "assignee"
        )

        if assignee_data:
            assignee_name = assignee_data.get(
                "displayName",
                "Unknown"
            )
        else:
            assignee_name = "Unassigned"


        results.append(
            f"{issue['key']} | "
            f"{summary} | "
            f"Status: {status_name} | "
            f"Priority: {priority_name} | "
            f"Assignee: {assignee_name}"
        )


    return "\n".join(results)