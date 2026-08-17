import os
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth


load_dotenv()

JIRA_URL = os.getenv("JIRA_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

PROJECT_KEY = "AAW"


url = f"{JIRA_URL}/rest/api/3/project/{PROJECT_KEY}"

response = requests.get(
    url,
    auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN),
    headers={
        "Accept": "application/json"
    }
)

print("Status code:", response.status_code)

if response.ok:
    data = response.json()

    print("Project name:", data.get("name"))
    print("Project key:", data.get("key"))
    print("Project ID:", data.get("id"))

else:
    print("Request failed:")
    print(response.text)