from langchain.tools import tool


@tool
def get_project_status(project_name: str) -> str:
    """
    Get the current status of a project.

    Args:
        project_name: Name of the project.
    """

    projects = {
        "Project X": {
            "status": "At Risk",
            "blockers": [
                "Authentication module is incomplete",
                "Database migration is delayed"
            ],
            "deadline": "September 15, 2026"
        }
    }

    project = projects.get(project_name)

    if project is None:
        return f"No information found for {project_name}."

    return str(project)