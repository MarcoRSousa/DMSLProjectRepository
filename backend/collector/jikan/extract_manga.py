"""
extract_manga.py.

Functionality: Provides helper functions for the extraction stage of the ETL pipeline (1st).

Methods:
    - get_last_visible_page(limit=25): Helper method for to determine last page given limit number of manga scraped (25 is max).
    - fetch_page(page, limit=25): Method that fetches a single page of manga entries via Jikan API (v4).
"""

import requests

# Define manga endpoint url; Define max manga to be scraped in the page.
BASE_URL = "https://api.jikan.moe/v4/manga"
PAGE_LIMIT = 25

# Create a persistent HTTP session.
session = requests.Session()

def get_last_visible_page(limit = PAGE_LIMIT) -> int:
    """Functionality: Determines the last_visible page of the manga endpoint using limit=25 by default."""

    # Specify the parameters
    params = {"page": 1, "limit": limit}

    try:
        # Try the request to fetch the page
        response = session.get(BASE_URL, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Error when querying the first page of manga endpoint."
        ) from e
    
    # Selects the scraped page data
    data = response.json()

    # Selects the pagination info
    pagination_info = data['pagination']

    # Return only the last visible page number
    return pagination_info['last_visible_page']


def fetch_page(page, limit=PAGE_LIMIT) -> list[dict]:
    """Functionality: Fetch a single specified page of manga."""

    # Specify the parameters
    params = {"page": page, "limit": limit}

    try:
        # Try the request to fetch the page
        response = session.get(BASE_URL, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to fetch page {page}."
        ) from e

    # Selects the scraped page data
    data = response.json()

    # Grabs manga data scraped from the page (returns list of manga dicts)
    manga_page = data['data']

    return manga_page
