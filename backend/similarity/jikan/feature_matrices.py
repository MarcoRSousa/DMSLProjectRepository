"""
Functionality: 
    Constructs feature matrices for manga series based on the features of genres, themes, authors, and serializations.
    Addionally constructs a feature matrix for the synopsis embeddings of manga series.
"""

from database.database import SessionLocal 
from database.models import Manga, Genres, Themes, Authors, Serializations
from paths import DATA_DIR
from paths import CONTENT_BASED_MANGA_TO_INDEX_NAME, CONTENT_BASED_INDEX_TO_MANGA_NAME

import numpy as np
from scipy.sparse import csr_matrix
import pickle


FEATURE_MODELS = {
    "genres": Genres,
    "themes": Themes,
    "authors": Authors,
    "serializations": Serializations,
}

FEATURE_COLUMNS = {
    "genres": Manga.genre_ids,
    "themes": Manga.theme_ids,
    "authors": Manga.author_ids,
    "serializations": Manga.serialization_ids,
}

EMBEDDING_DIM = 384  # Length of the embedding vector for the synopsis embeddings
ZERO_EMBEDDING = np.zeros(EMBEDDING_DIM,dtype=np.float32) # A zero vector for missing embeddings


def fetch_feature_mapping(feature_name: str) -> dict:
    """
    Functionality:
        Fetches the mapping of mal_id of the feature to its row number for the specified feature name (genre, theme, author, or serializations).
    """

    model = FEATURE_MODELS.get(feature_name)

    if model is None:
        raise ValueError(f"Unknown feature name: {feature_name}")

    session = SessionLocal()

    try:
        rows = (
            session.query(model.mal_id)
            .order_by(model.mal_id)
            .all()
        )
        return {row.mal_id: index for index, row in enumerate(rows)}
    finally:
        session.close()


def fetch_manga_feature_data(feature_name: str) -> list[dict]:
    """
    Functionality:
        Fetches the feature IDs for every manga series for the specified feature name (genre, theme, author, or serializations). 
        Returns one entry per manga, ordered by mal_id.
        Missing features are represented by an empty list.
    """

    session = SessionLocal()

    try:

        feature_column = FEATURE_COLUMNS.get(feature_name)

        rows = (
            session.query(
                Manga.mal_id,
                feature_column
            )
            .order_by(Manga.mal_id)
            .all()
        )

        return [
            {
                "mal_id": row.mal_id,
                f"{feature_name}_ids": getattr(row, feature_column.key) or []
            }
            for row in rows
        ]

    finally:
        session.close()


def construct_feature_matrix(feature_name: str) -> np.ndarray:
    """
    Functionality:
        Constructs a binary matrix for the specified feature name (genre, theme, author, or serializations) where each row represents a manga series and each column represents a unique feature ID.
        The matrix is filled with 1s and 0s indicating the presence or absence of the feature for each manga series in line with multihot encoding.

    Returns:
        np.ndarray: A binary matrix representing the presence of features for each manga series.
    """

    feature_mapping = fetch_feature_mapping(feature_name)
    manga_feature_data = fetch_manga_feature_data(feature_name)

    num_features = len(feature_mapping)
    num_manga = len(manga_feature_data)

    feature_matrix = np.zeros((num_manga, num_features), dtype=np.uint8)

    for i, manga in enumerate(manga_feature_data):
        for feature_id in manga[f"{feature_name}_ids"]:
                j = feature_mapping[feature_id]
                feature_matrix[i, j] = 1

    return feature_matrix

def construct_feature_matrix_sparse(feature_name: str):
    """
    Functinality:
        Constructs a csr representation of the multihot binary matrix for the specified feature name (genre, theme, author, or serializations).

    Note: This is done because the feature matrices are very sparse. So the CSR representaiton is far more efficient.
    """

    dense = construct_feature_matrix(feature_name)

    return csr_matrix(dense)


def construct_embedding_matrix() -> np.ndarray:
    """
    Functionality:
        Constructs a matrix for the synopsis embeddings of manga series where each row represents a manga series and each column represents a dimension of the embedding vector.
        Missing embeddings are represented by rows filled with zeros.

    Returns:
        np.ndarray: A matrix representing the synopsis embeddings for each manga series.
    """

    session = SessionLocal()

    try:

        rows = (
            session.query(
                Manga.mal_id,
                Manga.embedding
            )
            .order_by(Manga.mal_id)
            .all()
        )
        
        embedding_matrix = np.array([
            row.embedding if row.embedding is not None else ZERO_EMBEDDING for row in rows
            ], dtype=np.float32)

        return embedding_matrix

    finally:
        session.close()


def save_row_lookup() -> None:
    """
    Functionality:
        Saves the mapping between MAL IDs and matrix row indices for the content-based feature matrices.

        The mappings correspond to the row ordering used when constructing all content-based feature and embedding matrices.
    """

    session = SessionLocal()

    try:

        rows = (
            session.query(Manga.mal_id)
            .order_by(Manga.mal_id)
            .all()
        )

        content_based_manga_to_index = {}
        content_based_index_to_manga = {}

        for index, row in enumerate(rows):
            content_based_manga_to_index[row.mal_id] = index
            content_based_index_to_manga[index] = row.mal_id

        with open(DATA_DIR / CONTENT_BASED_MANGA_TO_INDEX_NAME, "wb") as f:
            pickle.dump(content_based_manga_to_index, f)

        with open(DATA_DIR / CONTENT_BASED_INDEX_TO_MANGA_NAME, "wb") as f:
            pickle.dump(content_based_index_to_manga, f)

    finally:
        session.close()

