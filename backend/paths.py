"""
Functionality: 
    Stores saving paths for data for reference elsewhere in the code.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).parent

CACHE_DIR = PROJECT_DIR / "similarity" / "cache"
CACHE_DIR.mkdir(exist_ok=True)

DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Mal_id to title mapping dictionary naming convention
MALID_TO_TITLE_NAME = 'mal_id_to_title.pkl'

# Collaborative Filtering mapping dictionaries naming convention
COLLABORATIVE_MANGA_TO_INDEX_NAME = 'collaborative_manga_to_index.pkl'
COLLABORATIVE_INDEX_TO_MANGA_NAME = 'collaborative_index_to_manga.pkl'

# Content Based mapping array naming convention
CONTENT_BASED_MANGA_TO_INDEX_NAME = 'content_based_manga_to_index.pkl'
CONTENT_BASED_INDEX_TO_MANGA_NAME = 'content_based_index_to_manga.pkl'

# Hybrid mapping array naming convention
HYBRID_MANGA_TO_INDEX_NAME = 'hybrid_manga_to_index.pkl'
HYBRID_INDEX_TO_MANGA_NAME = 'hybrid_index_to_manga.pkl'

# Similarity matrix naming convention
CONTENT_BASED_SIMILARITY_GRAPH_NAME = "content_based_similarity_knn_graph.npz"
COLLABORATIVE_SIMILARITY_GRAPH_NAME = "collaborative_similarity_knn_graph.npz"
HYBRID_SIMILARITY_GRAPH_NAME = "hybrid_similarity_knn_graph.npz"

# Name of similarity survey respondent results
RESPONDENT_DATA_NAME = 'respondent_data_by_mal_ids.csv'
