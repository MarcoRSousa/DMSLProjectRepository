"""
Functionality: Construct semantic embeddings for the synopsis of manga series using sentence transformers. Done in batches.
"""


from database.models import Manga
from database.database import SessionLocal
from sqlalchemy.dialects.postgresql import insert

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL)


def fetch_synopses() -> list[dict]:
    """
    Functionality:
        Retrieve the mal_id and synopsis of manga series using ORM from the database and return as a list of dictionaries.

    Returns:
        list: A list of dictionaries containing the mal_id and synopsis of each manga series.

    """
    session = SessionLocal()

    rows = (
        session.query(
            Manga.mal_id,
            Manga.synopsis
        )
        .filter(
            Manga.synopsis.isnot(None),
            Manga.embedding.is_(None)
            )
        .all()
    )

    session.close()

    return [
        {
            "mal_id": row.mal_id,
            "synopsis": row.synopsis
        }
        for row in rows
    ]


def embed_synopses(manga_rows: list[dict]) -> list[dict]:
    """
    Functionality:
        Generate embeddings for the synopsis of manga series in batches using the sentence transformer model.

    Args:
        manga_rows (list[dict]): A list of dictionaries containing the mal_id and synopsis of each manga series.

    Returns:
        list[dict]: A list of dictionaries containing the mal_id, and embedding of each manga series.

    """
    synopses = [row["synopsis"] for row in manga_rows]
    embeddings = model.encode(
        synopses,
        batch_size=64,
        show_progress_bar=True
    )

    embedded_rows = []

    for row, embedding in zip(manga_rows, embeddings):

        embedded_rows.append({
            "mal_id": row["mal_id"],
            "embedding": embedding.tolist()
        })

    return embedded_rows


def save_embeddings(embeddings: list[dict]) -> None:
    """
    Functionality:
        Save the generated embeddings for the synopsis of manga series to the manga table.

    Args:
        embeddings (list[dict]): A list of dictionaries containing the mal_id and embedding of each manga series.

    """
    session = SessionLocal()

    try:
        session.bulk_update_mappings(
            Manga,
            embeddings
        )
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def embeddings_etl():
    """
    Functionality:
        Fetches synposes, embeds them, and saves them to database.
    """

    try:
        manga_rows = fetch_synopses()
        embedded_rows = embed_synopses(manga_rows)
        save_embeddings(embedded_rows)
    except Exception as e:
        print(f"An error occurred: {e}")
    else:
        print("Semantic embedding of synopsis has finished successfully.")


if __name__ == "__main__":
    embeddings_etl()
