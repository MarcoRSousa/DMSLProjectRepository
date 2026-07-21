"""
ingest_manga.py.

Functionality: Directs the ETL pipeline to extract, transform, then load manga in pages (batches) into the Manga (Base) database model.

Methods:
    - ingest_manga(): Directs the ETL pipeline. Handles Jikan rate limit of 60 per minute.
"""

from time import sleep

from collector.jikan.extract_manga import get_last_visible_page, fetch_page
from collector.jikan.transform_manga import transform_page
from collector.jikan.load_manga import load_page

REQUEST_DELAY = 1.1


def ingest_manga(start_page: int = 1, end_page: int | None = None):
    """Directs the ETL pipeline. Handles Jikan rate limit of 60 requests/minute."""

    last_visible_page = get_last_visible_page()
    sleep(REQUEST_DELAY)

    if end_page is None:
        end_page = last_visible_page

    failed_pages = []

    for page in range(start_page, end_page + 1):

        print(f"Processing page {page}/{last_visible_page}...")

        try:
            # Extract
            manga_page = fetch_page(page)
            sleep(REQUEST_DELAY)

            # Transform
            formatted_manga_page, formatted_genre_page, formatted_theme_page, formatted_author_page, formatted_serialization_page = transform_page(manga_page)

            # Load
            load_page(formatted_manga_page, formatted_genre_page, formatted_theme_page, formatted_author_page, formatted_serialization_page)

        except Exception as e:
            print(f"Failed page {page}: {e}")
            failed_pages.append(page)
            continue

    print("\nManga ingestion completed.")

    if failed_pages:
        print(f"Failed pages ({len(failed_pages)}): {failed_pages}")
    else:
        print("All pages ingested successfully.")

if __name__ == "__main__":
    ingest_manga()
