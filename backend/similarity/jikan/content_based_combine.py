"""
Functionality:
    Conductor function which orchestrates the construction of similarity matrices for manga series based on their features and embeddings.
    This uses a row-wise approach to compute the similarity matrices, which is more memory efficient than computing the full similarity matrix at once.
    The function computes the cosine similarity for the synopsis embeddings and the Jaccard similarity for the binary feature matrices (genre, theme, author, and serializations).
"""

from similarity.jikan.feature_matrices import construct_feature_matrix_sparse, construct_embedding_matrix, save_row_lookup
from similarity.similarity_calculations import jaccard_similarity_row, cosine_similarity_batch
from paths import CACHE_DIR, DATA_DIR
from paths import CONTENT_BASED_SIMILARITY_GRAPH_NAME

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def save_feature_matrices_sparse():
    """
    Functionality:
        Saves feature matrices as CSR  for manga series as a .npz file in backend/similarity/cache/ for later use.
        Also saves the and embedding matrix as a .npy file.

    Note: This is done because the feature matrices are very sparse. So the CSR representaiton is far more efficient.
    """
    # Save each feature matrix
    for feature_name in ["genres", "themes", "authors", "serializations"]:
        sparse_matrix = construct_feature_matrix_sparse(feature_name)
        sp.save_npz(CACHE_DIR / f"{feature_name}_feature_matrix_csr.npz", sparse_matrix)


def save_embedding_matrix():
    """
    Functionality:
        Saves the and embedding matrix as a .npy file.
    """
    # Save the embedding matrix
    embedding_matrix = construct_embedding_matrix()
    np.save(CACHE_DIR / "embedding_matrix.npy", embedding_matrix)


def save_content_based_similarity_graph() -> None:
    """
    Functionality: 
        Computes the similarity of each manga by row using row-wise computations.
        Then saves the result in "knn_similarity_graph.npz" in /data.
        The saved structure has ["indices"] and ["similarities"]. 
        All indices, including each row's index, correspond to "matrix_mal_ids.npy" in /data
    """

    # Loads all 5 matrices
    genres_matrix = sp.load_npz(CACHE_DIR / "genres_feature_matrix_csr.npz")
    themes_matrix = sp.load_npz(CACHE_DIR / "themes_feature_matrix_csr.npz")
    authors_matrix = sp.load_npz(CACHE_DIR / "authors_feature_matrix_csr.npz")
    serializations_matrix = sp.load_npz(CACHE_DIR / "serializations_feature_matrix_csr.npz")
    embedding_matrix = np.load(CACHE_DIR / "embedding_matrix.npy", mmap_mode="r")

    # Determine and save the number of total manga (rows)
    num_manga_rows = embedding_matrix.shape[0]
    # Determine k for pruning (top k nearest neighbors)
    k = 100
    # Initialize for top k neighbor indices and similarities
    neighbor_indices = np.zeros((num_manga_rows,k), dtype=np.int32)
    neighbor_similarities = np.zeros((num_manga_rows,k), dtype=np.float32)
    # Define weights; alpha is weight for interation factor for serialization versus independent factor
    weights = np.array([0.20, 0.20, 0.05, 0.05, 0.50], dtype=np.float32)
    alpha = 0.75
    alpha_o = 1-alpha
    # Define a confidence score dependent on amount of information available for each manga
    manga_data_availability = np.zeros(num_manga_rows, dtype=np.float32)
    # Define manga norms and sizes
    genres_sizes = np.asarray(genres_matrix.sum(axis=1)).ravel()
    themes_sizes = np.asarray(themes_matrix.sum(axis=1)).ravel()
    authors_sizes = np.asarray(authors_matrix.sum(axis=1)).ravel()
    serializations_sizes = np.asarray(serializations_matrix.sum(axis=1)).ravel()
    embedding_norms = np.linalg.norm(embedding_matrix, axis=1)
    # Weight renormalization mask for each feature
    has_genres = genres_sizes > 0
    has_themes = themes_sizes > 0
    has_authors = authors_sizes > 0
    has_serializations = serializations_sizes > 0

    # Defining batch size to batch the embedding cosine computation
    BATCH_SIZE = 1024

    # Batched implementation for embedding
    for start_row in tqdm(range(0, num_manga_rows, BATCH_SIZE), desc="Content-Based Similarity Construction"):

        # Determine the end row
        end_row = min(start_row + BATCH_SIZE, num_manga_rows)

        # Calculate the similarity for the batch
        embedding_batch = cosine_similarity_batch(embedding_matrix, embedding_norms, start_row, end_row)

        # Row implementation for jaccard and features
        for local_row, manga_row in enumerate(range(start_row, end_row)):
            # Define the row
            genres_row = jaccard_similarity_row(genres_matrix, genres_sizes, manga_row)
            themes_row = jaccard_similarity_row(themes_matrix, themes_sizes, manga_row)
            authors_row = jaccard_similarity_row(authors_matrix, authors_sizes, manga_row)
            serializations_row = jaccard_similarity_row(serializations_matrix, serializations_sizes, manga_row)
            embedding_row = embedding_batch[:, local_row]

            # Weight renormalization
            available_weight_mask = np.array([
                has_genres[manga_row],
                has_themes[manga_row],
                has_authors[manga_row],
                has_serializations[manga_row],
                True
            ])
            available_weights = weights.copy()
            available_weights[~available_weight_mask] = 0
            # Determine row availability confidence
            manga_data_availability[manga_row] = available_weights.sum()
            # Renormalize
            available_weights /= available_weights.sum()

            # Perform similarity calculation
            weighted_row_similarity = (
                available_weights[0] * genres_row
                + available_weights[1] * themes_row
                + available_weights[2] * authors_row
                + available_weights[3] * serializations_row * ((alpha_o) + alpha*(genres_row + themes_row)/2)
                + available_weights[4] * embedding_row
            )
            
            # Set its own index to negative infinity so it doesn't relate to itself
            weighted_row_similarity[manga_row] = -np.inf

            # Pruning: Select the top k
            top_k_indices_row = np.argpartition(weighted_row_similarity, -k)[-k:]
            top_k_similarities_row = weighted_row_similarity[top_k_indices_row]

            # Sort with boolean mask (order) so first entry is top neighbor
            order = np.argsort(top_k_similarities_row)[::-1]
            top_k_indices_row = top_k_indices_row[order]
            top_k_similarities_row = top_k_similarities_row[order]

            # Store top indices and similarities for the row
            neighbor_indices[manga_row] = top_k_indices_row
            neighbor_similarities[manga_row] = top_k_similarities_row
    
    # Save the neighbor indices and similarities
    np.savez_compressed(
        DATA_DIR / CONTENT_BASED_SIMILARITY_GRAPH_NAME,
        indices=neighbor_indices,
        similarities=neighbor_similarities,
    )
    # Save the manga data availability as numpy array.
    np.save(
        DATA_DIR / "manga_data_availability.npy",
        manga_data_availability
    )


def content_combine():
    """
    Functionality: Performs the following steps:
        - Saves the feature matrices as csr matrices using save_feature_matrices_sparse().
        - Saves embedding matrix using save_embedding_matrix.
        - Saves the row lookup array to map the datbase mal_id to feature matrices using save_row_lookup().
        - Saves a similarity graph of top k nearest neighbors using save_content_based_similarity_graph().
        - Saves a data availability array using save_content_based_similarity_graph().
    """

    try:
        print("Saving feature matrices as sparse csr...")
        save_feature_matrices_sparse()
        print("Saving dense embedding feature matrix...")
        save_embedding_matrix()
        print("Saving content based mapping dictionaries...")
        save_row_lookup()
        print("Saving similarity matrices as top-k graph...")
        save_content_based_similarity_graph()
    except Exception as e:
        print(f"An error occurred: {e}")
    else:
        print("Feature matrices, row lookup matrix, similarity graph structure, and data availability have been saved successfully.")


if __name__ == "__main__":
    content_combine()
