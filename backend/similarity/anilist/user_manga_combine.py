"""
Functionality:
    Combines the construction of the 3 user-manga feature matrices and the calculation of similarity matrices for collaberative filtering.
"""

from similarity.anilist.user_manga_matrices import build_user_manga_matrix, build_user_status_interaction_matrix, build_user_favorite_matrix
from similarity.similarity_calculations import jaccard_similarity_batch_csr, cosine_similarity_batch_csr
from paths import CACHE_DIR, DATA_DIR
from paths import COLLABORATIVE_SIMILARITY_GRAPH_NAME

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def save_collaborative_feature_matrices_sparse():
    """
    Functionality:
        Saves feature matrices for anilist collaborative filtering CSR for manga series as a .npz file in backend/similarity/cache/.
        Specifically saves details about user-manga scores, user-status interaction weights, and user-favorite csr matrices.
    """

    build_user_manga_matrix()
    build_user_status_interaction_matrix()
    build_user_favorite_matrix()


def save_collaborative_similarity_graph() -> None:
    """
    Functionality: 
        Computes the similarity of each manga by using row-wise computations (in batches).
        Then saves the result in "knn_collab_similarity_graph.npz" in /data.
        The saved structure has ["indices"] and ["similarities"]. 
        All indices, including each row's index, correspond to "manga_to_index.pkl" and "user_to_index.pkl" in /data
    """
    
    # Loads all 3 user-item feature matrices
    user_manga_matrix = sp.load_npz(CACHE_DIR / "user_manga_matrix_csr.npz")
    user_interaction_matrix = sp.load_npz(CACHE_DIR / "user_interaction_matrix_csr.npz")
    user_favorite_matrix = sp.load_npz(CACHE_DIR / "user_favorite_matrix_csr.npz")

    # Perform transpose to have item-user feature matrices
    manga_user_matrix = user_manga_matrix.T
    interaction_user_matrix = user_interaction_matrix.T
    favorite_user_matrix = user_favorite_matrix.T

    # Determine and save the number of total manga (rows)
    num_manga_rows = manga_user_matrix.shape[0]
    # Determine k for pruning (top k nearest neighbors)
    k = 100
    # Initialize for top k neighbor indices and similarities
    neighbor_indices = np.zeros((num_manga_rows,k), dtype=np.int32)
    neighbor_similarities = np.zeros((num_manga_rows,k), dtype=np.float32)
    # Define weights
    weights = np.array([0.7, 0.25, 0.05], dtype=np.float32)
    # Define norms and sizes
    manga_user_norms = np.sqrt(manga_user_matrix.multiply(manga_user_matrix).sum(axis=1)).A1
    interaction_user_norms = np.sqrt(interaction_user_matrix.multiply(interaction_user_matrix).sum(axis=1)).A1
    favorite_user_sizes = np.asarray(favorite_user_matrix.sum(axis=1)).ravel()

    # Defining batch size
    BATCH_SIZE = 512

    for start_row in tqdm(range(0, num_manga_rows, BATCH_SIZE), desc="Collaborative Similarity Construction"):

        # Determine the end row
        end_row = min(start_row + BATCH_SIZE, num_manga_rows)
    
        # Calculate the similarity for the batch
        manga_user_batch = cosine_similarity_batch_csr(manga_user_matrix, manga_user_norms, start_row, end_row)
        interaction_user_batch = cosine_similarity_batch_csr(interaction_user_matrix, interaction_user_norms, start_row, end_row)
        favorite_user_batch = jaccard_similarity_batch_csr(favorite_user_matrix, favorite_user_sizes, start_row, end_row)

        # Perform the similarity calculation for the batch
        weighted_batch = (
            weights[0] * manga_user_batch
            + weights[1] * interaction_user_batch
            + weights[2] * favorite_user_batch
        ).astype(np.float32)

        # Go row by row to take the top k
        for local_row, manga_row in enumerate(range(start_row, end_row)):
          
            # Define the row
            weighted_row_similarity = weighted_batch[:, local_row].copy()

            # Remove the self similarity
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
        DATA_DIR / COLLABORATIVE_SIMILARITY_GRAPH_NAME,
        indices=neighbor_indices,
        similarities=neighbor_similarities,
    )

def collaborative_combine():
    """
    Functionality: Performs the following steps:
        - Saves the feature matrices as csr matrices using save_collaborative_feature_matrices_sparse().
        - Saves two user_to_index and manga_to_index lookups to /data using save_collaborative_feature_matrices_sparse().
        - Saves a similarity graph of top k nearest neighbors using save_collaborative_similarity_graph().
    """

    try:
        print("Saving feature matrices as sparse csr")
        save_collaborative_feature_matrices_sparse()
        print("Saving similarity matrices as slim top-k graph")
        save_collaborative_similarity_graph()
    except Exception as e:
        print(f"An error occurred: {e}")
    else:
        print("Collaborative filter feature matricies, manga_to_index lookup, and similarity graph has been saved successfully.")


if __name__ == "__main__":
    collaborative_combine()
