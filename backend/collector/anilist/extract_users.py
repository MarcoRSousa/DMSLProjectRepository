"""
Functionality: 
    Provides helper functions for the extraction stage of the anilist ETL pipeline (1st).

Note:
    The fetch method here fetches the top 5000 manga, and grabs all users that have reviewed those top 5000.
    Thus the collaborative filter only offers additonal information for users that review the popular manga.
"""

import requests
from time import sleep

# Define anilist bast endpoint url; Define max users to be scraped in the page. Define max retries and time between them.
BASE_URL = "https://graphql.anilist.co"
PAGE_LIMIT = 50
MAX_RETRIES = 5
RETRY_DELAY = 60
REQUEST_TIMEOUT = 60

# Create a persistent HTTP session.
session = requests.Session()


def fetch_user_page_by_popular_reviewed(page: int, limit: int = PAGE_LIMIT):
    """
    Functionality: Fetch the top 5000 most popular manga, and grabs all users that have reviewed those top 5000.
    """

    # Define the GraphQL query with variables
    query = '''
        query ($page: Int, $perPage: Int) {
            Page(page: $page, perPage: $perPage) {
                media (type: MANGA, sort: POPULARITY_DESC) {
                    reviews {
                    nodes {
                        user {
                            id
                            name
                        }
                        }
                    }
                }
            }
        }
     '''
    
    # Define variables
    variables = {
        "page": page,
        "perPage": limit
    }
    # Payload
    payload = {
        'query': query,
        'variables': variables
        }
    # Headers
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
        }

    try:
        # Try the request to fetch the page
        response = session.post(BASE_URL, json=payload, headers=headers)

       # Stop immediately and raise RuntimeError if rate limited
        if response.status_code == 429:
            raise RuntimeError(
                "AniList rate limit reached (HTTP 429). "
                "Stopping ingestion to avoid further requests."
            )
        response.raise_for_status()

        data = response.json()

    except requests.RequestException as e:
        raise RuntimeError(
            f"Query failed with status code {response.status_code}"
        ) from e

    return [
        review["user"]
        for media in data["data"]["Page"]["media"]
        for review in media["reviews"]["nodes"]
        if review["user"] is not None
    ]


def fetch_user_page_by_popular_recommended(page: int, limit: int = PAGE_LIMIT):
    """
    Functionality: Fetch the top 5000 most popular manga, and grabs all users that have recommended those top 5000.
    """

    # Define the GraphQL query with variables
    query = '''
        query Page($page: Int, $perPage: Int) {
            Page(page: $page, perPage: $perPage) {
                media (type: MANGA, sort: POPULARITY_DESC) {
                recommendations {
                    nodes {
                    user {
                        id
                        name
                    }
                    }
                }
                }
            }
        }
     '''
    
    # Define variables
    variables = {
        "page": page,
        "perPage": limit
    }
    # Payload
    payload = {
        'query': query,
        'variables': variables
        }
    # Headers
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
        }

    try:
        # Try the request to fetch the page
        response = session.post(BASE_URL, json=payload, headers=headers)

        # Stop immediately and raise RuntimeError if rate limited
        if response.status_code == 429:
            raise RuntimeError(
                "AniList rate limit reached (HTTP 429). "
                "Stopping ingestion to avoid further requests."
            )
        response.raise_for_status()

        data = response.json()

    except requests.RequestException as e:
        raise RuntimeError(
            f"Query failed with status code {response.status_code}"
        ) from e
    
    return [
        recommendation["user"]
        for media in data["data"]["Page"]["media"]
        for recommendation in media["recommendations"]["nodes"]
        if recommendation["user"] is not None
    ]


### CRAWL BY YEAR ###


def fetch_user_page_by_popular_reviewed_by_year(page: int, year: int = 1989, limit: int = PAGE_LIMIT) -> list[dict]:
    """
    Functionality: Fetch the top 5000 most popular manga for a given year, and grabs all users that have reviewed those top 5000.
    """

    # Define the GraphQL query with variables
    query = '''
    query Media($page: Int, $perPage: Int, $startDateGreater: FuzzyDateInt, $startDateLesser: FuzzyDateInt) {
        Page(page: $page, perPage: $perPage) {
            media(type: MANGA, sort: POPULARITY_DESC, startDate_greater: $startDateGreater, startDate_lesser: $startDateLesser) {
                reviews {
                    nodes {
                        user {
                            id
                            name
                        }
                    }
                }
            }
        }
    }
    '''

    # Define variables
    start_date = int(f"{year}0101")
    end_date = int(f"{year + 1}0101")
    variables = {
        "page": page,
        "perPage": limit,
        "startDateGreater": start_date,
        "startDateLesser": end_date,
    }
    # Payload
    payload = {
        'query': query,
        'variables': variables
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

            if response.status_code == 429:
                raise RuntimeError(
                    "AniList rate limit reached (HTTP 429). "
                    "Stopping ingestion."
                )
            if response.status_code == 504:
                raise requests.HTTPError(
                    "Gateway Timeout",
                    response=response,
                )

            response.raise_for_status()
            data = response.json()
            break

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
            raise
    else:
        raise RuntimeError(
            f"Failed after {MAX_RETRIES} retries."
        )

    return [
        review["user"]
        for media in data["data"]["Page"]["media"]
        for review in media["reviews"]["nodes"]
        if review["user"] is not None
    ]


def fetch_user_page_by_popular_recommended_by_year(page: int, year: int = 1989, limit: int = PAGE_LIMIT) -> list[dict]:
    """
    Functionality: Fetch the top 5000 most popular manga for a given year, and grabs all users that have recommended those top 5000.
    """

    # Define the GraphQL query with variables
    query = '''
    query Media($page: Int, $perPage: Int, $startDateGreater: FuzzyDateInt, $startDateLesser: FuzzyDateInt) {
        Page(page: $page, perPage: $perPage) {
            media(type: MANGA, sort: POPULARITY_DESC, startDate_greater: $startDateGreater, startDate_lesser: $startDateLesser) {
                recommendations {
                    nodes {
                        user {
                            id
                            name
                        }
                    }
                }
            }
        }
    }
     '''

    # Define variables
    start_date = int(f"{year}0101")
    end_date = int(f"{year + 1}0101")
    variables = {
        "page": page,
        "perPage": limit,
        "startDateGreater": start_date,
        "startDateLesser": end_date,
    }
    # Payload
    payload = {
        'query': query,
        'variables': variables
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

            if response.status_code == 429:
                raise RuntimeError(
                    "AniList rate limit reached (HTTP 429). "
                    "Stopping ingestion."
                )
            if response.status_code == 504:
                raise requests.HTTPError(
                    "Gateway Timeout",
                    response=response,
                )

            response.raise_for_status()
            data = response.json()
            break

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
            raise
    else:
        raise RuntimeError(
            f"Failed after {MAX_RETRIES} retries."
        )

    return [
        recommendation["user"]
        for media in data["data"]["Page"]["media"]
        for recommendation in media["recommendations"]["nodes"]
        if recommendation["user"] is not None
    ]
