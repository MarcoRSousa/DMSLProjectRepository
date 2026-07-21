"""
Functionality:
    Directs the ETL pipeline for user data.

NOTE: We need to recursively processes batches of users of users. We have to do recursion to avoid private users.
But we need to batch so that the API handles itself in a timely fashion because of rate limits.
"""

from time import sleep

from collector.anilist.extract_user_data import (
    extract_user_data_batch,
    PrivateUserBatchError,
)
from collector.anilist.transform_user_data import transform_user_batch_data
from collector.anilist.load_user_data import load_user_data_batch


from database.models import Anilist_Users
from database.database import SessionLocal

import math


BATCH_SIZE = 16


def process_user_batch(user_ids: list[int]) -> None:
    """
    Recursively processes a batch of users. This is done because many users are private.
    If a batch fails because of a private user, split it in half.
    """

    # Empty batch
    if not user_ids:
        return

    try:
        batch_user_data = extract_user_data_batch(user_ids)
        (
            user_scoring_data,
            user_favorites_data,
            user_manga_data,
        ) = transform_user_batch_data(batch_user_data)

        load_user_data_batch(
            user_scoring_data,
            user_favorites_data,
            user_manga_data,
        )

    except PrivateUserBatchError:
        # print(f"Private user detected in batch of {len(user_ids)}")
        if len(user_ids) == 1:
            print(f"Skipping private user {user_ids[0]}")
            return

        midpoint = len(user_ids) // 2

        left = user_ids[:midpoint]
        right = user_ids[midpoint:]

        # print(f"Splitting batch of {len(user_ids)} "f"into {len(left)} and {len(right)}")

        process_user_batch(left)
        process_user_batch(right)

    except RuntimeError:
        # Rate limit or fatal error.
        raise


def ingest_user_data(start_batch: int = 1,end_batch: int | None = None) -> None:
    """
    Functionality: 
        Directs the user ETL pipeline for user data. Ingests data for all users in user table.

    Note:
        Batch numbers naturally depend on batch size.
    """

    session = SessionLocal()

    try:
        rows = (
            session.query(Anilist_Users.user_id)
            .order_by(Anilist_Users.user_id)
            .all()
        )
    finally:
        session.close()

    all_user_ids =  [row.user_id for row in rows]

    if not all_user_ids:
        print("No users found.")
        return
    
    num_batches = math.ceil(len(all_user_ids) / BATCH_SIZE)

    if end_batch is None:
        end_batch = num_batches

    # Clamp to valid values
    start_batch = max(1, start_batch)
    end_batch = min(end_batch, num_batches)

    if start_batch > end_batch:
        raise ValueError("start_batch must be <= end_batch")

    failed_batches = []

    for batch_num, i in enumerate(range(0, len(all_user_ids), BATCH_SIZE), start=1):

        # Added to allow us to skip to wanted batches only for parameterization.
        if batch_num < start_batch:
            continue
        if batch_num > end_batch:
            break


        # Define a batch of user_ids to query
        user_ids_batch = all_user_ids[i:i + BATCH_SIZE]
        
        print(f"Processing batch {batch_num}/{num_batches}")

        try:
            process_user_batch(user_ids_batch)

        except RuntimeError:
            raise

        except Exception as e:

            print(f"Failed batch {batch_num}: {e}")

            failed_batches.append({
                "batch": batch_num,
                "first_user": user_ids_batch[0],
                "last_user": user_ids_batch[-1],
            })
        

    if failed_batches:
        print(f"Failed batches ({len(failed_batches)}): {failed_batches}")
    else:
        print("User data ingested successfully.")

if __name__ == "__main__":
    ingest_user_data(start_batch=372)
