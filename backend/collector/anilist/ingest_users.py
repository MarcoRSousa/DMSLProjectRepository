"""
Functionality: 
    Directs the ETL pipeline to extract and load users in pages (batches) into the Anilist_Users database model.
    Handles Anilist rate limit of 90 requests/minute.

    Anilist pagination limits to 5000 items.

"""

from time import sleep

from collector.anilist.extract_users import fetch_user_page_by_popular_reviewed_by_year, fetch_user_page_by_popular_recommended_by_year
from collector.anilist.transform_users import transform_users
from collector.anilist.load_users import load_users_batch

REQUEST_DELAY = 4


def ingest_users(start_year: int = 1989, end_year: int = 2025, start_batch_num: int = 1, end_batch_num: int = 100) -> None:
    """
    Functionality: 
        Directs the user ETL pipeline. Ingests users that have reviewesd or recommended manga in top (up to 5000) popularity for given year range.
        Handles Anilist rate limit of 90 requests/minute AND even 30 requests/minute under maintainence.
    """

    failed_review_pages = []
    failed_recommendation_pages = []

    for year in range(start_year, end_year + 1):
        
        print(f"Ingesting users related to manga released in the year {year}...")


        for batch_num in range(start_batch_num, end_batch_num + 1):

            print(f"Processing year {year}, page {batch_num}/{end_batch_num}...")

            try:
                # ETL for reviewed
                batch = fetch_user_page_by_popular_reviewed_by_year(batch_num, year)
                batch = transform_users(batch)
                load_users_batch(batch)
                # Delay to avoid API rate Limits
                sleep(REQUEST_DELAY)

            except RuntimeError:
                # Stop on seriously fatal error like Rate limit (429) or another fatal error.
                raise
            except Exception as e:
                # Otherwise note the erro but continue
                print(f"Failed page {batch_num}: {e}")
                failed_review_pages.append((batch_num, year))

            try:
                # ETL for recommended
                batch = fetch_user_page_by_popular_recommended_by_year(batch_num, year)
                batch = transform_users(batch)
                load_users_batch(batch)
                # Delay to avoid API rate Limits
                sleep(REQUEST_DELAY)

            except RuntimeError:
                # Stop on seriously fatal error like Rate limit (429) or another fatal error.
                raise
            except Exception as e:
                # Otherwise note the erro but continue
                print(f"Failed page {batch_num}: {e}")
                failed_recommendation_pages.append((batch_num, year))
            

    print("\nUser ingestion completed.")


    if failed_review_pages:
        print(
            f"Failed review pages ({len(failed_review_pages)}): "
            f"{failed_review_pages}"
        )
    if failed_recommendation_pages:
        print(
            f"Failed recommendation pages ({len(failed_recommendation_pages)}): "
            f"{failed_recommendation_pages}"
        )
    if not failed_review_pages and not failed_recommendation_pages:
        print("All users ingested successfully.")


if __name__ == "__main__":
    ingest_users(1989, 2025, 1, 4)
