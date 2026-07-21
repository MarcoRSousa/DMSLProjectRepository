"""
Functionality:
    Fetches mal_id and manga titles to a local static file to convert for recommender output.
"""

import pickle
from paths import DATA_DIR, MALID_TO_TITLE_NAME

from database.models import Manga
from database.database import SessionLocal


def save_manga_titles() -> None:
    session = SessionLocal()


    try:

        rows = (
            session.query(
                Manga.mal_id,
                Manga.title
            )
            .filter(
                Manga.title.isnot(None)
                )
            .all()
        )
    except:
        raise
    finally:
        session.close()

    malid_to_title = { row.mal_id: row.title for row in rows }

    with open(DATA_DIR / MALID_TO_TITLE_NAME, "wb") as f:
        pickle.dump(malid_to_title, f)


if __name__ == "__main__":
    save_manga_titles()
