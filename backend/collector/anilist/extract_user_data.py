"""
Functionality:
    Extracts user favorite manga data and reading/read data by status
"""


import requests
from time import sleep

BASE_URL = "https://graphql.anilist.co"
MAX_RETRIES = 5
RETRY_DELAY = 60
REQUEST_TIMEOUT = 60
REQUEST_DELAY = 5.0

# Create a persistent HTTP session.
session = requests.Session()


class PrivateUserBatchError(Exception):
    """
    Raised when a batch contains one or more private users.
    """
    pass


def extract_user_data_batch(user_ids: list[int]) -> dict:
    """
    Funcionality: 
        Extracts manga reading and favorite information for the input user_ids.
        Does so by combining a single query for one user into a combined query.

        NOTE: Handles Anilist rate limit of 90 requests/minute AND even 30 requests/minute under maintainence.
        Generally this is handled in ingest but I did it here for regardless of if failure occurs or not.
    """

    # This is a container for all the queries when constructed together
    bundled_query = "query {"

    for user_id in user_ids:

        # Bundling the queries by concatenating
        bundled_query += f"""
            user_{user_id}: MediaListCollection(userId: {user_id}, type: MANGA) {{
                user {{
                    favourites {{
                    manga {{
                        nodes {{
                            idMal
                        }}
                    }}
                }}
                mediaListOptions {{
                    scoreFormat
                }}
            }}
            lists {{
                status
                entries {{
                    media {{
                        idMal
                    }}
                    score
                    progress
                    }}
                }}
            }}
        """

    # Finish the exterior of the query bundle
    bundled_query += '}'

    # Payload
    payload = {
        'query': bundled_query
        }
    # Headers
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
        }

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = session.post(
                BASE_URL,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            # Stop immediately on rate limiting
            if response.status_code == 429:
                raise RuntimeError(
                    "AniList rate limit reached (HTTP 429). "
                    "Stopping ingestion."
                )

            # Retry temporary server failures
            if response.status_code in {502, 503, 504}:
                raise requests.HTTPError(
                    f"Temporary server error ({response.status_code})",
                    response=response,
                )

            data = response.json()

            # Handle Private User specifically.
            if response.status_code == 404:

                if any(
                    error.get("message") == "Private User"
                    for error in data.get("errors", [])
                ):
                    raise PrivateUserBatchError()

            # Other HTTP errors.
            response.raise_for_status()

            return data
        except PrivateUserBatchError:
            raise
        except requests.HTTPError as e:
            status = (
                e.response.status_code
                if e.response is not None
                else None
            )
            if status in {502, 503, 504}:

                print(
                    f"Temporary server error ({status}). "
                    f"Retry {attempt}/{MAX_RETRIES} "
                    f"in {RETRY_DELAY} seconds..."
                )

                sleep(RETRY_DELAY)
                continue

            raise RuntimeError(
                f"Query failed (HTTP {status})."
            ) from e
        except (
            requests.ConnectionError,
            requests.Timeout,
        ):

            print(
                f"Connection issue. "
                f"Retry {attempt}/{MAX_RETRIES} "
                f"in {RETRY_DELAY} seconds..."
            )
            sleep(RETRY_DELAY)

        finally:
            sleep(REQUEST_DELAY)

    else:
        raise RuntimeError(
            f"Failed after {MAX_RETRIES} retries."
        )
