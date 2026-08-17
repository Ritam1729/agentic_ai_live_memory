import os
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth


load_dotenv()

JIRA_URL = os.getenv("JIRA_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")


url = f"{JIRA_URL}/rest/api/3/myself"

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

    print("Authentication successful!")
    print("Display name:", data.get("displayName"))
    print("Account ID:", data.get("accountId"))

else:
    print("Authentication failed.")
    print(response.text)