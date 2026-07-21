"""
Functionality:
    Cleans user data from the table anilist_ser_data.
"""

from sqlalchemy import delete, update

from database.database import SessionLocal
from database.models import Anilist_Users, Anilist_User_Manga

MAX_SCORE = {
    "POINT_100": 100,
    "POINT_10_DECIMAL": 10,
    "POINT_10": 10,
    "POINT_5": 5,
    "POINT_3": 3,
}


def remove_null_no_progress() -> None:
    """
    Functionality:
        Removes rows where status IS NULL and progress = 0.
    """
    
    session = SessionLocal()

    try:

        user_manga_data_cleanup_statement = delete(Anilist_User_Manga).where(
            Anilist_User_Manga.status.is_(None), Anilist_User_Manga.progress == 0
            )

        rows_deleted = session.execute(user_manga_data_cleanup_statement)
        session.commit()

        print(f"Deleted {rows_deleted.rowcount} rows.")
        
    except:
        raise
    finally:
        session.close()

def update_user_status_null() -> None:
    """
    Functionality:
        Changes rows having a NULL status to "OTHER".
    """

    session = SessionLocal()

    try:

        user_manga_data_null_update_statement = (
            update(Anilist_User_Manga)
            .where(Anilist_User_Manga.status.is_(None))
            .values(status="OTHER")
        )

        rows_changed = session.execute(user_manga_data_null_update_statement)
        session.commit()

        print(f"Updated {rows_changed.rowcount} rows.")

    except:
        session.rollback()
        raise

    finally:
        session.close()

def update_user_scores() -> None:
    """
    Functionality:
        User scores are 0 when not scored. This method changes those to NULL/None.
    """

    session = SessionLocal()

    try:

        user_manga_score_update_statement = (
            update(Anilist_User_Manga)
            .where(Anilist_User_Manga.score == 0)
            .values(score=None)
        )

        rows_changed = session.execute(user_manga_score_update_statement)
        session.commit()

        print(f"Updated {rows_changed.rowcount} rows.")

    except:
        session.rollback()
        raise

    finally:
        session.close()

def normalize_user_scores() -> None:
    """
    Functionality:
        Computes normalized scores on a 0-100 scale while preserving the original AniList score.
    """

    session = SessionLocal()

    try:

        rows = (
            session.query(
                Anilist_User_Manga,
                Anilist_Users.score_type,
            )
            .join(
                Anilist_Users,
                Anilist_User_Manga.user_id == Anilist_Users.user_id,
            )
            .all()
        )

        for manga, score_type in rows:

            if manga.score is None:
                continue

            max_score = MAX_SCORE.get(score_type)

            if max_score is None:
                continue

            manga.score_normalized = (
                (manga.score - 1)
                / (max_score - 1)
            ) * 100

        session.commit()

    except:
        session.rollback()
        raise

    finally:
        session.close()

def clean_user_data() -> None:
    """
    Functionality:
        Cleans user data from the table anilist_user_manga.
    """

    remove_null_no_progress()
    update_user_status_null()
    update_user_scores()
    normalize_user_scores()
    
    print("User data cleaned.")


if __name__ == "__main__":
    clean_user_data()
