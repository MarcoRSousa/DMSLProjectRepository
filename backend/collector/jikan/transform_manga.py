"""
transform_manga.py.

Functionality: Provides helper functions for the transforming stage of the ETL pipeline (2nd).

Methods:
    - transform_manga(jikan_manga): Transforms a single manga with columns found in manga model in models.py.
    - transform_page(jikan_manga_page): Transforms a page (batch) of manga using transform_manga.
    - clean_synopsis(text): Cleans a Jikan synopsis by removing metadata that is not part of the story description.
"""


import re


def clean_synopsis(text: str | None) -> str | None:
    if text is None:
        return None

    if text.strip() == "No synopsis has been added for this series yet.":
        return None
    
    # Define patterns to remove from the synopsis
    patterns = [
        r"\[Written by MAL Rewrite\]",
        r"\(Source:[^)]+\)",
        r"Included one-shots?:.*",
        r"Note:.*",
    ]

    # Remove patterns from the text
    for pattern in patterns:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

    # Normalize whitespace: replace multiple spaces with a single space.
    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    # Remove multiple consecutive newlines and trailing whitespace
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = text.strip()

    return text if text else None


def transform_manga(jikan_manga) -> dict:
    """ 
    Functionality: 
    - Transforms a single manga with columns found in manga model in models.py.
    - Additionally constructs theme and genre pages for loading into their respective tables.
    Note: details_fetched is a flag for when we fetch additional details using the id later. Initialized as false here.
    """

    formatted_manga = {
        "mal_id" : jikan_manga['mal_id'],

        "title" : jikan_manga['title'],
        "title_english" : jikan_manga['title_english'],

        "status" : jikan_manga['status'],
        "chapters" : jikan_manga['chapters'],
        "volumes" : jikan_manga['volumes'],

        "synopsis" : clean_synopsis(jikan_manga['synopsis']),
        "background" : jikan_manga['background'],

        "author_ids" : [author['mal_id'] for author in jikan_manga['authors']],
        "serialization_ids" : [serialization['mal_id'] for serialization in jikan_manga['serializations']],

        "genre_ids" : [genre['mal_id'] for genre in jikan_manga['genres']],
        "theme_ids" : [theme['mal_id'] for theme in jikan_manga['themes']],

        # "score" : jikan_manga['score'],
        # "scored_by" : jikan_manga['scored_by'],
        # "rank" : jikan_manga['rank'],
        # "popularity" : jikan_manga['popularity'],

        "embedding" : None,

        "details_fetched": False
    }

    return formatted_manga


def transform_page(jikan_manga_page) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """
    Functionality: Transforms a page (batch) of manga, genres, themes, authors, and serializations using transform_manga.
    """

    formatted_manga_page = []
    formatted_theme_page = {}
    formatted_genre_page = {}
    formatted_author_page = {}
    formatted_serialization_page = {}

    for manga in jikan_manga_page:
        # Transform manga
        formatted_manga_page.append(transform_manga(manga))
        # Transform genres
        for genre in manga['genres']:
            formatted_genre_page[genre['mal_id']] = {
                "mal_id": genre['mal_id'],
                "name": genre['name'],
                "type": genre['type']
            }
        # Transform themes
        for theme in manga['themes']:
            formatted_theme_page[theme['mal_id']] = {
                "mal_id": theme['mal_id'],
                "name": theme['name'],
                "type": theme['type']
            }
        # Transform authors
        for author in manga['authors']:
            formatted_author_page[author['mal_id']] = {
                "mal_id": author['mal_id'],
                "name": author['name'],
                "type": author['type']
            }
        # Transform serializations
        for serialization in manga['serializations']:
            formatted_serialization_page[serialization['mal_id']] = {
                "mal_id": serialization['mal_id'],
                "name": serialization['name'],
                "type": serialization['type']
            }
            
    return (formatted_manga_page, list(formatted_genre_page.values()), list(formatted_theme_page.values()), list(formatted_author_page.values()), list(formatted_serialization_page.values()))
