"""
Functionality:
    Combines the collaborative and content-based graph into a hybrid_graph for recommendations.
    Constructs new index and manga mapping dictionaries for the new similarity graph.

Assumption:
    Currently assumes both the content combine and the collaborative combine has already taken place.
"""

import numpy as np
import pickle
from tqdm import tqdm

from paths import DATA_DIR
from paths import COLLABORATIVE_SIMILARITY_GRAPH_NAME, COLLABORATIVE_INDEX_TO_MANGA_NAME, COLLABORATIVE_MANGA_TO_INDEX_NAME
from paths import CONTENT_BASED_SIMILARITY_GRAPH_NAME, CONTENT_BASED_INDEX_TO_MANGA_NAME, CONTENT_BASED_MANGA_TO_INDEX_NAME
from paths import HYBRID_SIMILARITY_GRAPH_NAME, HYBRID_INDEX_TO_MANGA_NAME, HYBRID_MANGA_TO_INDEX_NAME


def hybrid_combine(content_weight=0.3, collaborative_weight=0.7, k=100):
    """
    Functionality:
        Combines the collaborative and content-based graph into a hybrid_graph for recommendations.
        Constructs new index and manga mapping dictionaries for the new similarity graph.
    """

    # Load the data
    knn_content_based_graph = np.load(DATA_DIR / CONTENT_BASED_SIMILARITY_GRAPH_NAME)
    knn_collab_based_graph = np.load(DATA_DIR / COLLABORATIVE_SIMILARITY_GRAPH_NAME)
    # Access indices and similarities in line
    all_content_based_indices = knn_content_based_graph["indices"]
    content_based_similarities = knn_content_based_graph["similarities"]
    all_collab_indices = knn_collab_based_graph["indices"]
    collab_similarities = knn_collab_based_graph["similarities"]

    # Load the mappings
    with open(DATA_DIR / CONTENT_BASED_MANGA_TO_INDEX_NAME, "rb") as f:
        manga_to_index_content = pickle.load(f)
    with open(DATA_DIR / CONTENT_BASED_INDEX_TO_MANGA_NAME, "rb") as f:
        index_to_manga_content = pickle.load(f)
    with open(DATA_DIR / COLLABORATIVE_MANGA_TO_INDEX_NAME, "rb") as f:
        manga_to_index_collab = pickle.load(f)
    with open(DATA_DIR / COLLABORATIVE_INDEX_TO_MANGA_NAME, "rb") as f:
        index_to_manga_collab = pickle.load(f)

    # Construct a union of the all possible ids and construct mapping dictionaries
    all_mal_ids = sorted(
        set(index_to_manga_content.values())
        | set(index_to_manga_collab.values())
    )
    manga_to_index_hybrid = {
        mal_id: i for i, mal_id in enumerate(all_mal_ids)
    }
    index_to_manga_hybrid = {
        i: mal_id for i, mal_id in enumerate(all_mal_ids)
    }

    # Define neighbor indices and similarities up to k
    num_manga = len(all_mal_ids)
    neighbor_indices = np.full(
        (num_manga, k),
        -1,
        dtype=np.int32,
    )
    neighbor_similarities = np.zeros(
        (num_manga, k),
        dtype=np.float32,
    )

    for hybrid_row, mal_id in tqdm(enumerate(all_mal_ids),  total=num_manga, desc="Constructing Hybrid Graph"):

        neighbor_scores = {}

        # Handle content row score computation
        content_row = manga_to_index_content.get(mal_id)

        if content_row is not None:

            content_neighbor_indices = all_content_based_indices[content_row]
            content_neighbor_similarities = content_based_similarities[content_row]

            for neighbor_idx, sim in zip(content_neighbor_indices, content_neighbor_similarities):

                # Handle the event we for a k too large
                if neighbor_idx == -1:
                    continue

                neighbor_mal = index_to_manga_content[neighbor_idx]

                neighbor_scores[neighbor_mal] = (
                    neighbor_scores.get(neighbor_mal, 0.0)
                    + content_weight * sim
                )

        # Handle collab row score computation
        collab_row = manga_to_index_collab.get(mal_id)

        if collab_row is not None:

            collab_neighbor_indices = all_collab_indices[collab_row]
            collab_neighbor_similarities = collab_similarities[collab_row]

            for neighbor_idx, sim in zip(collab_neighbor_indices, collab_neighbor_similarities):

                # Handle the event we for a k too large
                if neighbor_idx == -1:
                    continue
                
                neighbor_mal = index_to_manga_collab[neighbor_idx]

                neighbor_scores[neighbor_mal] = (
                    neighbor_scores.get(neighbor_mal, 0.0)
                    + collaborative_weight * sim
                )

        # Remove itself
        neighbor_scores.pop(mal_id, None)

        # Sort
        sorted_neighbors = sorted(
            neighbor_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        # Convert back to indices and similarities structure
        hybrid_indices = [
            manga_to_index_hybrid[neighbor_mal]
            for neighbor_mal, _ in sorted_neighbors
        ]
        hybrid_sims = [
            sim
            for _, sim in sorted_neighbors
        ]

        # Store (and also prune at the same time - preventing an input k too large)
        actual_k = min(k, len(sorted_neighbors))
        neighbor_indices[hybrid_row, :actual_k] = hybrid_indices
        neighbor_similarities[hybrid_row, :actual_k] = hybrid_sims

    
    # Save the neighbor indices and similarities
    np.savez_compressed(
        DATA_DIR / HYBRID_SIMILARITY_GRAPH_NAME,
        indices=neighbor_indices,
        similarities=neighbor_similarities,
    )
    # Save the mapping dictionaries...
    with open(DATA_DIR / HYBRID_MANGA_TO_INDEX_NAME, "wb") as f:
        pickle.dump(manga_to_index_hybrid, f)
    with open(DATA_DIR / HYBRID_INDEX_TO_MANGA_NAME, "wb") as f:
        pickle.dump(index_to_manga_hybrid, f)


if __name__ == "__main__":
    hybrid_combine()
