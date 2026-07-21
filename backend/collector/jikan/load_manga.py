"""
load_manga.py.

Functionality: Provides helper functions for the loading stage of the ETL pipeline (3rd).

Methods:
    - load_manga_page(formatted_manga_page, formatted_genre_page, formatted_theme_page): 
    Loads a manga page in bulk into the Manga table. Also loads Themes and Genres into their respective tables.
"""

from sqlalchemy.dialects.postgresql import insert

from database.models import Manga, Genres, Themes, Authors, Serializations
from database.database import SessionLocal


def load_page(formatted_manga_page, formatted_genre_page, formatted_theme_page, formatted_author_page, formatted_serialization_page):

    session = SessionLocal()

    try:

        session.execute(
            insert(Manga),
            formatted_manga_page
        )

        genres_statement = (
            insert(Genres)
            .values(formatted_genre_page)
            .on_conflict_do_nothing(
                index_elements=["mal_id"]
            )
        )
        session.execute(genres_statement)

        themes_statement = (
            insert(Themes)
            .values(formatted_theme_page)
            .on_conflict_do_nothing(
                index_elements=["mal_id"]
            )
        )
        session.execute(themes_statement)

        authors_statement = (
            insert(Authors)
            .values(formatted_author_page)
            .on_conflict_do_nothing(
                index_elements=["mal_id"]
            )
        )
        session.execute(authors_statement)

        serializations_statement = (
            insert(Serializations)
            .values(formatted_serialization_page)
            .on_conflict_do_nothing(
                index_elements=["mal_id"]
            )
        )
        session.execute(serializations_statement)

        session.commit()

    except:
        session.rollback()
        raise

    finally:
        session.close()
