"""
Functionality:
    Trains and evaluates the performance of the recommenders.
"""

import numpy as np
import pandas as pd
from collections import Counter

from recommendations.recommender import RecommendationType, recommend_from_manga
from similarity.hybrid.hybrid_combine import hybrid_combine
from paths import DATA_DIR, RESPONDENT_DATA_NAME


def load_survey_responses() -> pd.DataFrame:
    """
    Loads the survey responses used for recommendation evaluation.
    """

    return pd.read_csv(DATA_DIR / RESPONDENT_DATA_NAME)

def remove_no_response(df: pd.DataFrame):
    """
    Functionality:
        Removes the rows in which a user did not respond.
        Assumes the first response column (st1) is blank if no responses.
    """

    return df.dropna(subset=['st1']).copy()

def construct_precision_recall_structure(df: pd.DataFrame) -> dict[list[set]]:
    """
    Functionality:
        Constructs a dictionary where each key is a target manga id and vlaues are list of ids for each respondent's suggested manga.
    """

    # Define suggested title columns
    st_cols = ['st1', 'st2', 'st3', 'st4', 'st5']

    result_dict = (
        df.assign(
            st_set=df[st_cols].apply(
                lambda row: set(row.dropna()),
                axis=1
            )
        )
        .groupby("mal_id")["st_set"]
        .apply(list)
        .to_dict()
    )

    return result_dict

def construct_ndcg_structure(df: pd.DataFrame) -> dict[int, dict[int, int]]:
    """
    Functionality:
        Constructs a dictionary where each key is a target manga and each value is a dictionary mapping suggested manga IDs to the number of respondents who suggested them.
    
    Returns:
        {
            target_mal_id: {
                suggested_mal_id: frequency,
                ...
            },
            ...
        }
    """

    st_cols = ["st1", "st2", "st3", "st4", "st5"]

    ndcg_dict = {}

    for target_mal_id, group in df.groupby("mal_id"):

        counter = Counter()

        for _, row in group.iterrows():
            suggestions = row[st_cols].dropna().astype(int)
            counter.update(suggestions)

        ndcg_dict[target_mal_id] = dict(counter)

    return ndcg_dict


def precision_recall_at_k(recommendation_type: RecommendationType, target_mal_id: int, ground_truth_sets: list[set[int]], k: int = 5) -> tuple[float,float]:
    """
    Functionality:
        Computes the AVERAGE Precision@k AND Recall@k across all respondents for a single target manga.
    Returns:
        tuple[average precisions, average recalls]
    """

    recommendations = set(
        recommend_from_manga(
            recommendation_type,
            target_mal_id,
            n=k
        )
    )

    precisions = []
    recalls = []

    for relevant in ground_truth_sets:

        overlap = len(recommendations & relevant)

        precisions.append(overlap / k)
        recalls.append(overlap / len(relevant))

    return (
        np.mean(precisions),
        np.mean(recalls),
    )

def dcg(relevances: list[float]) -> float:
    """
    Functionality:
        Computes the Discounted Cumulative Gain (DCG) for an orderedlist of relevance scores.
    Formula:
        DCG@k = sum_i=1^k rel_i / log_2(i+1)
    """

    return sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(relevances)
    )

def ndcg_at_k(recommendation_type: RecommendationType, target_mal_id: int, relevance_dict: dict[int, int], k: int = 5) -> float:
    """
    Functionality:
        Computes NDCG@k for a single target manga.

    Args:
        - recommendation_type:Recommendation model to evaluate.
        - target_mal_id: Query manga.
        - relevance_dict: Dictionary mapping manga IDs to relevance scores.
        - k: Number of recommendations.

    Returns:
        NDCG@k.
    """

    recommendations = recommend_from_manga(
        recommendation_type,
        target_mal_id,
        n=k,
    )

    # Relevance of the recommended manga
    predicted_relevances = [
        relevance_dict.get(mal_id, 0)
        for mal_id in recommendations
    ]

    dcg_score = dcg(predicted_relevances)

    # Construct the ideal ranking
    ideal_relevances = sorted(
        relevance_dict.values(),
        reverse=True
    )[:k]

    idcg_score = dcg(ideal_relevances)

    if idcg_score == 0:
        return 0.0

    return dcg_score / idcg_score



if __name__ == "__main__":

    print("Evaluation takes place on a grid of 11 weight distributions...")
    # Load the data; Structure for use
    df = load_survey_responses()
    df_clean = remove_no_response(df)
    precision_recall_dict = construct_precision_recall_structure(df_clean)
    ndcg_dict = construct_ndcg_structure(df_clean)

    # Define weights 
    content_weights = np.linspace(0, 1, 11)

    # Manga to test and number for @k
    at_k = 8
    # The tested target manga
    manga_mal_ids = [2, 113138, 13, 112268, 35243, 90125]

    # Array to store precision, recall, and ndcg data in
    precisions = np.zeros((len(manga_mal_ids), len(content_weights)))
    recalls = np.zeros((len(manga_mal_ids), len(content_weights)))
    ndcgs = np.zeros((len(manga_mal_ids), len(content_weights)))

    for j, weight in enumerate(content_weights):

        # Recompute the weight
        hybrid_combine(weight, 1-weight, 100)

        for i, mal_id in enumerate(manga_mal_ids):
            precision, recall = precision_recall_at_k(RecommendationType.HYBRID, mal_id, precision_recall_dict[mal_id], at_k)
            ndcg_score = ndcg_at_k(RecommendationType.HYBRID, mal_id, ndcg_dict[mal_id], at_k)

            precisions[i,j] = np.round(precision,3)
            recalls[i,j] = np.round(recall, 3)
            ndcgs[i,j] = np.round(ndcg_score, 3)

    weight_labels = [f"{w:.1f}" for w in content_weights]

    precision_df = pd.DataFrame(
        precisions,
        index=manga_mal_ids,
        columns=weight_labels,
    )

    recall_df = pd.DataFrame(
        recalls,
        index=manga_mal_ids,
        columns=weight_labels,
    )

    ndcg_df = pd.DataFrame(
        ndcgs,
        index=manga_mal_ids,
        columns=weight_labels,
    )

    # Reset combine to default weight
    hybrid_combine()

    print("Precision Table:")
    print(precision_df)
    print('\n')
    print("Recall Table:")
    print(recall_df)
    print('\n')
    print("NDCG Table:")
    print(ndcg_df)
    print('\n')
    
    mean_precision = np.round(precision_df.mean(axis=0), 3)
    mean_recall = np.round(recall_df.mean(axis=0),3)
    mean_ndcg = np.round(ndcg_df.mean(axis=0),3)
    print("Averages: ")
    print("Precision Averages:")
    print(mean_precision)
    print("Recall Averages:")
    print(mean_recall)
    print("NDCG Averages:")
    print(mean_ndcg)
    print("\n")

    print("respondents per manga:")
    respondants_per_manga = df_clean.groupby("mal_id").size()
    print(respondants_per_manga)
