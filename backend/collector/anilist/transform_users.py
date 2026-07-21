"""
Functionality:
    2nd step in user ETL pipeline. Clean user batches.
"""


def transform_users(users: list[dict]) -> list[dict]:
    IGNORE_USERS = {"AniBot", "Removed"}

    return [
        {
            "user_id": user["id"],
            "name": user["name"]
        }
        for user in users
        if user["name"] not in IGNORE_USERS
    ]
