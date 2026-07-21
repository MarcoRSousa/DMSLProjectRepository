"""
Functionality:
    Computes the similarity for a given row of a feature matrix against all other rows in the matrix.
TODO: 
- Change jaccard simiarity row to handle batches here and in combine.
- Make everything in terms of csr so I can combine both functions and simply use the SAME two batch functions for both sides.
"""

import numpy as np


### JIKAN IMPLEMENTATION - CONTENT-BASED SIMILARITY CALCULATION ###


def jaccard_similarity_row(feature_matrix_csr, row_sizes: np.ndarray, query_index: int) -> np.ndarray:
    """
    Functionality:
        Computes the Jaccard similarity for a given row of the binary feature matrix against all other rows.

    Args:
        feature_matrix (scipy csr matrix): CSR matrix form of the feature matrix being a binary matrix where each row represents a manga series and each column represents a unique feature ID.
        row_sizes (np.ndarray): A vector containing the sizes of each row in the feature matrix.
        query_index (int): The index of the *row* for which to compute similarities.

    Returns:
        np.ndarray: A row vector representing the Jaccard similarities between the given manga series and all other manga series.
    """

    query_vector = feature_matrix_csr.getrow(query_index)

    intersection = (feature_matrix_csr @ query_vector.T).toarray().ravel()
    union = row_sizes[query_index] + row_sizes - intersection

    # Handle division by zero by using np.divide with the 'where' parameter to avoid NaN values in the similarity matrix.
    similarity_row = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union != 0
        )

    return similarity_row


def cosine_similarity_batch(embedding_matrix: np.ndarray, embedding_norms: np.ndarray,start_row: int, end_row: int,) -> np.ndarray:
    """
    Functionality:
        Computes the cosine similarity for batches. Used for the jikan content-based filtering.
    """

    batch = embedding_matrix[start_row:end_row]
    numerator = embedding_matrix @ batch.T

    query_norms = embedding_norms[start_row:end_row]
    denominator = (embedding_norms[:, None]*query_norms[None, :])

    similarity_batch = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator != 0,
    )

    return similarity_batch


### ANILIST IMPLEMENTATION - COLLABORATIVE-FILTERING BASED SIMILARITY CALCULATION ###


def cosine_similarity_batch_csr(matrix_csr, norms: np.ndarray, start_row: int, end_row: int,) -> np.ndarray:
    """
    Functionality:
        Computes the cosine similarity for batches. 
        Currently used for the anilist content-based filtering.
    """

    batch = matrix_csr[start_row:end_row]
    numerator = (matrix_csr @ batch.T).toarray()

    query_norms = norms[start_row:end_row]
    denominator = (norms[:, None]*query_norms[None, :])

    similarity_batch = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator != 0,
    )

    return similarity_batch


def jaccard_similarity_batch_csr(matrix_csr, row_sizes: np.ndarray, start_row: int, end_row: int) -> np.ndarray:
    """
    Functionality:
        Computes the Jaccard similarity between a batch of rows and all rows of CSR matrix.
        Currently used for the anilist content-based filtering.
    """

    batch = matrix_csr[start_row:end_row]

    # Number of shared features/users
    intersection = (matrix_csr @ batch.T).toarray()

    # Number of features/users in each query row
    query_sizes = row_sizes[start_row:end_row]

    # |A| + |B| - |A ∩ B|
    union = (
        row_sizes[:, None]
        + query_sizes[None, :]
        - intersection
    )

    similarity_batch = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union != 0,
    )

    return similarity_batch
