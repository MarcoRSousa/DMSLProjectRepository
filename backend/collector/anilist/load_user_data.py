"""
Functionality:
    Load user data into the three anilist tables.
Note:
    To update, we might consider .on_conflict_do_update(...), such as changes to user progress.
"""

from sqlalchemy.dialects.postgresql import insert

from database.database import SessionLocal
from database.models import Anilist_Users, Anilist_User_Favorites, Anilist_User_Manga


def load_user_data_batch(user_scoring_data: list[dict], user_favorites_data: list[dict], user_manga_data: list[dict]) -> None:
    """
    Functionality:
        Load user data into the three anilist tables.
        The user table is updated with available scoring_type. The other two tables are populated.
    """

    session = SessionLocal()

    try:
        # Update user table
        session.bulk_update_mappings(
            Anilist_Users,
            user_scoring_data
        )
        session.commit()

        # populate user favourites table
        favorites_insert_statement = (
            insert(Anilist_User_Favorites)
            .values(user_favorites_data)
            .on_conflict_do_nothing(
                index_elements=["user_id", "mal_id"]
            )
        )

        manga_insert_statement = (
            insert(Anilist_User_Manga)
            .values(user_manga_data)
            .on_conflict_do_nothing(
                index_elements=["user_id", "mal_id"]
            )
        )

        session.execute(favorites_insert_statement)
        session.execute(manga_insert_statement)

        session.commit()
        
    except:
        session.rollback()
        raise
    finally:
        session.close()
