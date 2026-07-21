"""
Functionality:
    Constructs user-manga csr matrix to next use for similarity measurement.
"""

from database.database import SessionLocal 
from database.models import Anilist_User_Manga, Anilist_User_Favorites
from paths import CACHE_DIR, DATA_DIR
from paths import COLLABORATIVE_MANGA_TO_INDEX_NAME, COLLABORATIVE_INDEX_TO_MANGA_NAME

import pickle
import numpy as np
from scipy.sparse import csr_matrix, save_npz

# Weight distribution of user interaction value for a manga
status_interaction_weights = {
    "PLANNING": 0.075,
    "CURRENT": 0.15,
    "PAUSED": 0.05,
    "COMPLETED": 0.25,
    "DROPPED": 0.025,
    "REPEATING": 0.3,
    "OTHER": 0.15,
}


def build_user_manga_matrix() -> None:
    """
    Functionality:
        Builds user-manga matrix with each row being a user and each column being a manga. Each value is their normalized score for a respective manga.
        Also saves user feature mapping.
    """

    # Feature mapping.  Will be used to map indices for users and manga
    user_to_index = {}
    manga_to_index = {}

    # CSR Matrix Data
    row_indices = []
    col_indices = []
    data = []

    # Session start
    session = SessionLocal()

    try:
        # Grab batches of rows using yield_per
        rows = (
            session.query(
                Anilist_User_Manga.user_id,
                Anilist_User_Manga.mal_id,
                Anilist_User_Manga.score_normalized,
            )
            .filter(
                Anilist_User_Manga.score_normalized.is_not(None)
            )
            .yield_per(50000)
        )

        for row in rows:
            # Build the mapping index and grab index at same time using setdefault
            user_idx = user_to_index.setdefault(
                row.user_id,
                len(user_to_index),
            )

            manga_idx = manga_to_index.setdefault(
                row.mal_id,
                len(manga_to_index),
            )

            # CSR Matrix Data
            row_indices.append(user_idx)
            col_indices.append(manga_idx)
            data.append(row.score_normalized)

            if len(data) % 200000 == 0:
                print(
                    f"Processed {len(data):,} interactions..."
                )
    except:
        raise
    finally:
        session.close()


    # Reverse the manga_to_index dictionary
    index_to_manga = {
        index: mal_id
        for mal_id, index in manga_to_index.items()
    }
    # Construct the CSR Matrix
    row_indices = np.array(row_indices, dtype=np.int32)
    col_indices = np.array(col_indices, dtype=np.int32)
    data = np.array(data, dtype=np.float32)

    user_item_matrix = csr_matrix(
        (
            data,
            (row_indices, col_indices),
        ),
        shape=(
            len(user_to_index),
            len(manga_to_index),
        ),
    )
    # Save User-Item CSR Matrix
    save_npz(CACHE_DIR / f"user_manga_matrix_csr.npz", user_item_matrix)
    # Save the mapping dictionaries...
    with open(CACHE_DIR / "user_to_index.pkl", "wb") as f:
        pickle.dump(user_to_index, f)
    with open(DATA_DIR / COLLABORATIVE_MANGA_TO_INDEX_NAME, "wb") as f:
        pickle.dump(manga_to_index, f)
    with open(DATA_DIR / COLLABORATIVE_INDEX_TO_MANGA_NAME, "wb") as f:
        pickle.dump(index_to_manga, f)

    # Report statistics

    if user_to_index and manga_to_index:
        density = len(data) / (
            len(user_to_index) * len(manga_to_index)
        )
    else:
        density = 0

    print(f"Users: {len(user_to_index):,}")
    print(f"Manga: {len(manga_to_index):,}")
    print(f"Interactions: {len(data):,}")
    print(f"Density: {density:.6%}")
    print(f"Average interactions per user: "f"{len(data) / len(user_to_index):.2f}")

def build_user_status_interaction_matrix() -> None:
    """
    Functionality:
        Builds user-manga matrix with each row being a user and each column being manga. Each item is a score associated with a status weight for a respective manga.
    """

    # Load user and manga index mapping
    with open(CACHE_DIR / "user_to_index.pkl", "rb") as f:
        user_to_index = pickle.load(f)
    with open(DATA_DIR / COLLABORATIVE_MANGA_TO_INDEX_NAME, "rb") as f:
        manga_to_index = pickle.load(f)

    # CSR Matrix Data
    row_indices = []
    col_indices = []
    data = []

    # Session start
    session = SessionLocal()

    try:
        # Grab batches of rows using yield_per
        rows = (
            session.query(
                Anilist_User_Manga.user_id,
                Anilist_User_Manga.mal_id,
                Anilist_User_Manga.status,
            )
            .yield_per(50000)
        )

        for row in rows:

            user_idx = user_to_index.get(row.user_id)
            manga_idx = manga_to_index.get(row.mal_id)

            if user_idx is None or manga_idx is None:
                continue
            
            # Convert the status to a score:
            status_score = status_interaction_weights.get(row.status, 0.0)

            # CSR Matrix Data
            row_indices.append(user_idx)
            col_indices.append(manga_idx)
            data.append(status_score)

            if len(data) % 200000 == 0:
                print(
                    f"Processed {len(data):,} interactions..."
                )
    except:
        raise
    finally:
        session.close()
        
    # Construct the CSR Matrix
    row_indices = np.array(row_indices, dtype=np.int32)
    col_indices = np.array(col_indices, dtype=np.int32)
    data = np.array(data, dtype=np.float32)

    user_interaction_matrix = csr_matrix(
        (
            data,
            (row_indices, col_indices),
        ),
        shape=(
            len(user_to_index),
            len(manga_to_index),
        ),
    )
    # Save User-Item CSR Matrix
    save_npz(CACHE_DIR / f"user_interaction_matrix_csr.npz", user_interaction_matrix)

def build_user_favorite_matrix() -> None:
    """
    Functionality:
        Builds user-manga matrix with each row being a user and each column being manga. Each item is a score 1 for a respective favorite manga or otherwise 0.
    """

    # Load user and manga index mapping
    with open(CACHE_DIR / "user_to_index.pkl", "rb") as f:
        user_to_index = pickle.load(f)
    with open(DATA_DIR / COLLABORATIVE_MANGA_TO_INDEX_NAME, "rb") as f:
        manga_to_index = pickle.load(f)

    # CSR Matrix Data
    row_indices = []
    col_indices = []
    data = []

    # Session start
    session = SessionLocal()

    try:
        # Grab batches of rows using yield_per
        rows = (
            session.query(
                Anilist_User_Favorites.user_id,
                Anilist_User_Favorites.mal_id,
            )
            .yield_per(20000)
        )

        for row in rows:

            user_idx = user_to_index.get(row.user_id)
            manga_idx = manga_to_index.get(row.mal_id)

            if user_idx is None or manga_idx is None:
                continue

            # CSR Matrix Data. Append 1 if its a favorite
            row_indices.append(user_idx)
            col_indices.append(manga_idx)
            data.append(1)

            if len(data) % 20000 == 0:
                print(
                    f"Processed {len(data):,} interactions..."
                )
    except:
        raise
    finally:
        session.close()
        
    # Construct the CSR Matrix
    row_indices = np.array(row_indices, dtype=np.int32)
    col_indices = np.array(col_indices, dtype=np.int32)
    data = np.array(data, dtype=np.int8)

    user_favorite_matrix = csr_matrix(
        (
            data,
            (row_indices, col_indices),
        ),
        shape=(
            len(user_to_index),
            len(manga_to_index),
        ),
    )
    # Save User-Item CSR Matrix
    save_npz(CACHE_DIR / f"user_favorite_matrix_csr.npz", user_favorite_matrix)
