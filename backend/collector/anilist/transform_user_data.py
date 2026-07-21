"""
Functionality:
    Transforms user data into usable data for inserrtion/loading into user tables.
"""


def transform_user_batch_data(data: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Functionality:
        Transforms user data into usable data for inserrtion/loading into user tables.

    Returns:
        - user_scoring_data: Contains the user id and their score type to update the anilist_user table.
        - user_favorites_data: Contains all the user's favorited data to update the anilist_user_favourites table.
        - user_manga_data: Contains data about the user's read or planning to read series to update the anilist_user_manga table.
    """

    user_favorites_data = []
    user_manga_data = []
    user_scoring_data = []

    for alias, user_data in data["data"].items():

        user_id = int(alias.split("_")[1])
        score_type = user_data["user"]["mediaListOptions"]["scoreFormat"]

        user_scoring_data.append({
            "user_id":user_id,
            "score_type": score_type
        })


        favorites_manga_data = user_data["user"]["favourites"]["manga"]["nodes"]

        for favorite_manga in favorites_manga_data:

            id_mal = favorite_manga["idMal"]
            if id_mal is not None:

                formatted_fav_data = {
                    "user_id": user_id,
                    "mal_id": id_mal
                }
                user_favorites_data.append(formatted_fav_data)


        manga_list_data = user_data["lists"]

        for single_list in manga_list_data:

            status = single_list["status"]

            for manga_entry in single_list["entries"]:

                mal_id = manga_entry["media"]["idMal"]

                if mal_id is None:
                    continue

                formatted_manga_data = {
                    "user_id": user_id,
                    "mal_id": mal_id,
                    "status": status,
                    "score": manga_entry["score"],
                    "progress": manga_entry["progress"]
                }
                user_manga_data.append(formatted_manga_data)

    return user_scoring_data, user_favorites_data, user_manga_data
