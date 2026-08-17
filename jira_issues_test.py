import os
import requests

from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth


load_dotenv()

JIRA_URL = os.getenv("JIRA_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

PROJECT_KEY = "AAW"


jql = f"project = {PROJECT_KEY} ORDER BY created DESC"

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


print("Status code:", response.status_code)


if response.ok:

    data = response.json()

    issues = data.get("issues", [])

    print(f"\nFound {len(issues)} issues\n")

    for issue in issues:

        fields = issue["fields"]

        summary = fields.get("summary")
        status = fields.get("status", {}).get("name")
        priority = fields.get("priority")

        if priority:
            priority = priority.get("name")

        assignee = fields.get("assignee")

        if assignee:
            assignee = assignee.get("displayName")
        else:
            assignee = "Unassigned"

        print(
            f"{issue['key']} | "
            f"{summary} | "
            f"Status: {status} | "
            f"Priority: {priority} | "
            f"Assignee: {assignee}"
        )

else:

    print("Request failed:")
    print(response.text)