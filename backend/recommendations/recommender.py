"""
Functionality: 
    Holds reccomender functions which recommend manga.

TODO:
    - Preload data so each query doesn't reload all the data.
    - A large refactor would move everything to interacting with databases, but was kept local static files for project submission.
"""

import numpy as np
import pickle

from enum import Enum

from paths import DATA_DIR
from paths import COLLABORATIVE_SIMILARITY_GRAPH_NAME, COLLABORATIVE_INDEX_TO_MANGA_NAME, COLLABORATIVE_MANGA_TO_INDEX_NAME
from paths import CONTENT_BASED_SIMILARITY_GRAPH_NAME, CONTENT_BASED_INDEX_TO_MANGA_NAME, CONTENT_BASED_MANGA_TO_INDEX_NAME
from paths import HYBRID_SIMILARITY_GRAPH_NAME, HYBRID_INDEX_TO_MANGA_NAME, HYBRID_MANGA_TO_INDEX_NAME
from paths import MALID_TO_TITLE_NAME


class RecommendationType(Enum):
    """
    Handles switching between different recommendation types content-based, collaborative, hybrid.
    """
    CONTENT = "content"
    COLLABORATIVE = "collaborative"
    HYBRID = "hybrid"


def load_similarity_data(recommendation_type: RecommendationType):
    """
    Loads in the data related to the input RecommendationType
    """
    if recommendation_type == RecommendationType.CONTENT:
        # Load the data
        knn_content_based_graph = np.load(DATA_DIR / CONTENT_BASED_SIMILARITY_GRAPH_NAME)
        # Access indices and similarities in line
        all_content_based_indices = knn_content_based_graph["indices"]
        content_based_similarities = knn_content_based_graph["similarities"]

        # Load the mappings
        with open(DATA_DIR / CONTENT_BASED_MANGA_TO_INDEX_NAME, "rb") as f:
            manga_to_index_content = pickle.load(f)
        with open(DATA_DIR / CONTENT_BASED_INDEX_TO_MANGA_NAME, "rb") as f:
            index_to_manga_content = pickle.load(f)

        return all_content_based_indices, content_based_similarities, manga_to_index_content, index_to_manga_content
    
    elif recommendation_type == RecommendationType.COLLABORATIVE:
        # Load the data
        knn_collaborative_graph = np.load(DATA_DIR / COLLABORATIVE_SIMILARITY_GRAPH_NAME)
        # Access indices and similarities
        all_collaborative_indices = knn_collaborative_graph["indices"]
        collaborative_similarities = knn_collaborative_graph["similarities"]

        # Load the mappings
        with open(DATA_DIR / COLLABORATIVE_MANGA_TO_INDEX_NAME, "rb") as f:
            manga_to_index_collab = pickle.load(f)
        with open(DATA_DIR / COLLABORATIVE_INDEX_TO_MANGA_NAME, "rb") as f:
            index_to_manga_collab = pickle.load(f)

        return all_collaborative_indices, collaborative_similarities, manga_to_index_collab, index_to_manga_collab
    
    elif recommendation_type == RecommendationType.HYBRID:
        # Load the data
        knn_hybrid_graph = np.load(DATA_DIR / HYBRID_SIMILARITY_GRAPH_NAME)
        # Access indices and similarities
        all_hybrid_indices = knn_hybrid_graph["indices"]
        hybrid_similarities = knn_hybrid_graph["similarities"]

        # Load the mappings
        with open(DATA_DIR / HYBRID_MANGA_TO_INDEX_NAME, "rb") as f:
            manga_to_index_hybrid = pickle.load(f)
        with open(DATA_DIR / HYBRID_INDEX_TO_MANGA_NAME, "rb") as f:
            index_to_manga_hybrid = pickle.load(f)

        return all_hybrid_indices, hybrid_similarities, manga_to_index_hybrid, index_to_manga_hybrid
    else:
        raise ValueError(
            f"Unknown recommendation type: {recommendation_type}"
        )


def recommend_from_manga(recommendation_type: RecommendationType, given_mal_id: int, n: int = 5) -> list[int]:
    """
    Functionality:
        Performs recommendation. Takes a single manga and finds n most similar neighbors.
        Assumes indices were sorted from highest similarity to lowest.

    Args:
        - recommendation_type: RecommendationType class which selects CONTENT COLLABORATEIVE, or HYBRID data
        - given_mal_id: mal_id of a given manga
        - n: The number of suggested manga

    Return:
        Returns an array of mal_ids of recommended manga.
    """

    # Load data
    all_indices, _, manga_to_index, index_to_manga = load_similarity_data(recommendation_type)

    # Acquire the index corresponding to the input mal_id
    row = manga_to_index.get(given_mal_id)
    # Validation
    if row is None:
        raise ValueError(
            f"Manga {given_mal_id} is not available for the given recommendation type."
            )

    # Acquire the neighbor indices
    neighbors = all_indices[row][:n]
    # Convert neighbor indices to neighbor mal_ids
    neighbor_mal_ids = [index_to_manga[neighbor] for neighbor in neighbors]

    return neighbor_mal_ids


def recommend_from_manga_list_uniform_average(recommendation_type: RecommendationType, given_mal_ids: list[int], n: int = 5) -> list[int]:
    """
    Functionality:
        Performs recommendation. Takes a list of manga and finds n most similar neighbors by averaging with equal weights.
    
    Notes:
        Note that the averaging taken here punishes items which do not appear as often because it takes average for all indices, not JUST among present indices.

    Args:
        - recommendation_type: RecommendationType class which selects CONTENT COLLABORATEIVE, or HYBRID data
        - given_mal_ids: list of mal_ids of given manga
        - n: The number of suggested manga

    Return:
        Returns an array of mal_ids of recommended manga.
    """

    # Load data
    all_indices, similarities, manga_to_index, index_to_manga = load_similarity_data(recommendation_type)

    # Validate
    for mal_id in given_mal_ids:
        row = manga_to_index.get(mal_id)

        if row is None:
            raise ValueError(
                f"Manga {mal_id} is not available for the given recommendation type."
            )
    # Acquire the indices corresponding to the input mal_id
    input_rows =  [manga_to_index[mal_id] for mal_id in given_mal_ids]

    scores = {}

    for input_row in input_rows:

        # For the given manga, select the neighbor indices and similarities
        neighbors = all_indices[input_row]
        sims = similarities[input_row]

        # For each neighbor and its similarity, add to the similarity score
        for neighbor, sim in zip(neighbors, sims):
            scores[neighbor] = scores.get(neighbor, 0.0) + sim

    # Compute the unweighted average
    for neighbor in scores:
        scores[neighbor] /= len(input_rows)

    # Remove rows that have already recommendation manga. Return None otherwise (do nothing).
    for input_row in input_rows:
        scores.pop(input_row, None)
    
    # Sort
    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Return 
    return np.array([index_to_manga[index] for index, score in sorted_scores[:n]], dtype=np.int32)


def recommend_from_manga_list_weighted(recommendation_type: RecommendationType, given_mal_ids: list[int], weights: list[float], n: int = 5) -> list[int]:
    """
    Functionality:
        Performs recommendation. Takes a list of manga and finds n most similar neighbors by averaging with specified weights.
        Assumes indices were sorted from highest similarity to lowest.
        
    Notes:
        Note that the averaging taken here punishes items which do not appear as often because it takes average for all indices, not JUST among present indices.

    Args:
        - recommendation_type: RecommendationType class which selects CONTENT COLLABORATEIVE, or HYBRID data
        - given_mal_ids: list of mal_ids of given manga
        - weights: list of weights
        - n: The number of suggested manga

    Return:
        Returns an array of mal_ids of recommended manga.
    """
    # Validate the weights
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError(
            "The weights must sum to 1."
        )
    if len(weights) != len(given_mal_ids):
        raise ValueError(
            "The number of weights must equal the number of manga."
        )
    
    # Load data
    all_indices, similarities, manga_to_index, index_to_manga = load_similarity_data(recommendation_type)

    # Validate
    for mal_id in given_mal_ids:
        row = manga_to_index.get(mal_id)

        if row is None:
            raise ValueError(
                f"Manga {mal_id} is not available for the given recommendation type."
            )
    # Acquire the indices corresponding to the input mal_id
    input_rows =  [manga_to_index[mal_id] for mal_id in given_mal_ids]

    scores = {}

    for i, input_row in enumerate(input_rows):

        # For the given manga, select the neighbor indices and similarities
        neighbors = all_indices[input_row]
        sims = similarities[input_row]

        # For each neighbor and its similarity, add to the similarity score using the respective manga weight.
        for neighbor, sim in zip(neighbors, sims):
            scores[neighbor] = scores.get(neighbor, 0.0) + (weights[i] * sim)

    # Remove rows that have already recommendation manga. Return None otherwise (do nothing).
    for input_row in input_rows:
        scores.pop(input_row, None)
    
    # Sort
    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Return 
    return np.array([index_to_manga[index] for index, score in sorted_scores[:n]], dtype=np.int32)


def recommend_from_manga_list_weighted_by_category(recommendation_type: RecommendationType, given_categories_of_mal_ids: list[list[int]], weights: list[float], n: int = 5) -> list[int]:
    """
    Functionality:
        Performs recommendation. Takes a list of lists of manga and finds n most similar neighbors by averaging with weights corresponding to each category.
        Assumes indices were sorted from highest similarity to lowest on backend.
        
    Notes:
        Note that the averaging taken here punishes items which do not appear as often because it takes average for all indices, not JUST among present indices.
    
    Args:
        - recommendation_type: RecommendationType class which selects CONTENT COLLABORATEIVE, or HYBRID data
        - given_categories_of_mal_ids: list of lists of ints containing mal_id for each category. Each category is a list.
        - weights: Weights of each category. The index should correspong to the index of the list (category) in given_categories_of_mal_ids.
        - n: The number manga recommended.
    Return:
        Returns an array of mal_ids of recommended manga.
    """
    # Validate the weights
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError(
            "The weights must sum to 1."
        )
    if len(weights) != len(given_categories_of_mal_ids):
        raise ValueError(
            "The number of weights must equal the number of categories."
        )
    
    # Load data
    all_indices, similarities, manga_to_index, index_to_manga = load_similarity_data(recommendation_type)

    input_categories = []

    # Acquire the indices corresponding to the input mal_id
    for category_row in given_categories_of_mal_ids:
        # Validate to prevent divide by zero error
        if len(category_row) == 0:
            raise ValueError(
                "Categories must contain at least one manga."
            )
        # Validate the existance of each manga mal_id
        for mal_id in category_row:
            row = manga_to_index.get(mal_id)

            if row is None:
                raise ValueError(
                    f"Manga {mal_id} is not available for the given recommendation type."
                )
        # Acquire the indices corresponding to the input mal_id
        input_rows =  [manga_to_index[mal_id] for mal_id in category_row]
        
        input_categories.append(input_rows)
    
    scores = {}

    # For every category
    for i, category_indices in enumerate(input_categories):

        # Define the weight of the category (normalized so more manga in a category has equal influence)
        weight = weights[i] / len(category_indices)

        # For every manga (of each category)
        for input_row in category_indices:

            # For the given manga, select the neighbor indices and similarities
            neighbors = all_indices[input_row]
            sims = similarities[input_row]

            # For each neighbor and its similarity, add to the similarity score
            for neighbor, sim in zip(neighbors, sims):
                scores[neighbor] = scores.get(neighbor, 0.0) + (weight * sim)

    # Remove rows that have already recommendation manga. Return None otherwise (do nothing).
    flat_categories = np.concatenate(input_categories, dtype=np.int32)
    for input_row in flat_categories:
        scores.pop(input_row, None)
    
    # Sort
    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Return 
    return np.array([index_to_manga[index] for index, score in sorted_scores[:n]], dtype=np.int32)


def mal_ids_to_titles(mal_ids: list[int])->list[str]:
    """
    Functionality:
        Helper method to convert list of mal_ids to list of titles.
    Note:
        Uses a static local dictionary file rather than table lookup.
    """

    with open(DATA_DIR / MALID_TO_TITLE_NAME, "rb") as f:
        mal_id_to_title = pickle.load(f)

    return [mal_id_to_title[mal_id] for mal_id in mal_ids]


if __name__ == "__main__":

    print("Note that most similar to least similar goes from left to right.")
    print("\n")
    print("The following compares the output when suggesting a SINGLE MANGA (Berserk):")

    print("CONTENT-BASED FILTERING")
    top_10_simple = recommend_from_manga(RecommendationType.CONTENT, 2, 8)
    print(mal_ids_to_titles(top_10_simple))

    print("COLLABORATIVE FILTERING")
    top_10_simple = recommend_from_manga(RecommendationType.COLLABORATIVE, 2, 8)
    print(mal_ids_to_titles(top_10_simple))

    print("HYBRID")
    top_10_simple = recommend_from_manga(RecommendationType.HYBRID, 2, 8)
    print(mal_ids_to_titles(top_10_simple))

    print("\n")

    print("The following compares the output when suggesting more than one Manga with specified weights (0.7~Berserk AND 0.3~Claymore):")

    print("CONTENT-BASED FILTERING")
    top_10_weighted = recommend_from_manga_list_weighted(RecommendationType.CONTENT,[2, 583], [0.7,0.3])
    print(mal_ids_to_titles(top_10_weighted))

    print("COLLABORATIVE FILTERING")
    top_10_weighted = recommend_from_manga_list_weighted(RecommendationType.COLLABORATIVE,[2, 583], [0.7,0.3])
    print(mal_ids_to_titles(top_10_weighted))

    print("HYBRID")
    top_10_weighted = recommend_from_manga_list_weighted(RecommendationType.HYBRID,[2, 583], [0.7,0.3])
    print(mal_ids_to_titles(top_10_weighted))

    print("\n")

    print("The following recommends for categories of manga and uses weights on those categories. this allows users to get recommendations to their portfolio.")
    print("Category 1: Berserk and Violence Jack ~ 0.7. Claymore and Vinland Saga ~ 0.3")
    top_10_cat = recommend_from_manga_list_weighted_by_category(RecommendationType.HYBRID,[[2, 1442],[583, 642]], [0.7,0.3], 8)
    print(mal_ids_to_titles(top_10_cat))

    print("\n")

    print("The following shows the single manga output for the hybrid model fr each of the evaluation targets using content-collaborative weights 0.3-0.7")
    print("\n")
    print("Berserk")
    top_10_simple = recommend_from_manga(RecommendationType.HYBRID, 2, 8)
    print(mal_ids_to_titles(top_10_simple))
    print("JJK:")
    top_10_simple = recommend_from_manga(RecommendationType.HYBRID, 113138, 8)
    print(mal_ids_to_titles(top_10_simple))
    print("One Piece")
    top_10_simple = recommend_from_manga(RecommendationType.HYBRID, 13, 8)
    print(mal_ids_to_titles(top_10_simple))
    print("MDD")
    top_10_simple = recommend_from_manga(RecommendationType.HYBRID, 112268, 8)
    print(mal_ids_to_titles(top_10_simple))
    print("Haikyu!!")
    top_10_simple = recommend_from_manga(RecommendationType.HYBRID, 35243, 8)
    print(mal_ids_to_titles(top_10_simple))
    print("Kaguya")
    top_10_simple = recommend_from_manga(RecommendationType.HYBRID, 90125, 8)
    print(mal_ids_to_titles(top_10_simple))
