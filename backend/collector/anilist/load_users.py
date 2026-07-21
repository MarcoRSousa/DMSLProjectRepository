"""
Functionality:
    Load anilist users into the user database.
"""

from sqlalchemy.dialects.postgresql import insert

from database.database import SessionLocal
from database.models import Anilist_Users


def load_users_batch(list_of_users: list[dict]) -> None:
    """
    Functionality:
        Loads a batch of AniList users into the database. Existing users are ignored.
    """

    session = SessionLocal()

    try:
        statement = (
            insert(Anilist_Users)
            .values(list_of_users)
            .on_conflict_do_nothing(
                index_elements=["user_id"]
            )
        )

        session.execute(statement)
        session.commit()

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()
