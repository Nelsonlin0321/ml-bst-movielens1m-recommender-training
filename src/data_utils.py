import re
import pandas as pd
from zipfile import ZipFile
from urllib.request import urlretrieve
from typing import Dict, Tuple, Union, AnyStr
from sklearn.preprocessing import MinMaxScaler


def download_data() -> None:
    urlretrieve(
        "http://files.grouplens.org/datasets/movielens/ml-1m.zip", "./movielens.zip")
    ZipFile("movielens.zip", "r").extractall()


def read_source_data(dir_path="ml-1m") -> Tuple[pd.DataFrame]:
    users = pd.read_csv(
        f"{dir_path}/users.dat",
        sep="::",
        names=["user_id", "sex", "age_group", "occupation", "zip_code"],
        engine="python",
    )

    ratings = pd.read_csv(
        f"{dir_path}/ratings.dat",
        sep="::",
        names=["user_id", "movie_id", "rating", "unix_timestamp"],
        engine="python",
    )

    movies = pd.read_csv(
        f"{dir_path}/movies.dat",
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="ISO-8859-1",
    )

    return users, ratings, movies


def encode_id(df, col) -> Dict[Union[AnyStr, int], int]:
    ids = df[df[col].notnull()][col].unique().tolist()
    ids = sorted(ids)
    id_map_dict = {x: i + 1 for i, x in enumerate(ids)}
    id_map_dict["UNK"] = 0

    df[f"{col}_index"] = df[col].fillna("UNK").map(id_map_dict)

    return id_map_dict


def encode_sex(users: pd.DataFrame):
    sex_id_map_dict = {"M": 0.0, "F": 1.0, "UNK": 0.5}
    users["sex"] = users["sex"].map(sex_id_map_dict)
    return sex_id_map_dict


def encode_genres(movies: pd.DataFrame, max_genres_length: int) -> None:
    genres_set = set()
    for genres in movies["genres"]:
        genres_split = genres.split("|")
        genres_set.update(genres_split)

    movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))

    genres_map_dict = {x: i + 1 for i, x in enumerate(sorted(genres_set))}
    genres_map_dict["UNK"] = 0

    movies["genres_ids"] = movies["genres"].apply(
        lambda x: [genres_map_dict[g] for g in x])

    max_genres_length = 4
    movies["genres_ids"] = movies["genres_ids"].apply(
        lambda x: (x + [genres_map_dict["UNK"]] *
                   max_genres_length)[:max_genres_length]
    )

    return genres_map_dict


def encode_rating(ratings: pd.DataFrame) -> MinMaxScaler:
    min_max_scaler = MinMaxScaler()
    ratings["norm_rating"] = min_max_scaler.fit_transform(
        ratings["rating"].values.reshape(-1, 1)
    )[:, 0]

    return min_max_scaler


def create_sequences(values, window_size, step_size):
    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            break
        sequences.append(seq)
        start_index += step_size
    return sequences


def generate_sequence_data(input_sequences_data, sequence_length=2):
    step_size = 1
    output_sequences_data = input_sequences_data.copy()
    output_sequences_data.movie_sequence = output_sequences_data.movie_sequence.apply(
        lambda ids: create_sequences(ids, sequence_length, step_size)
    )
    output_sequences_data.genres_ids_sequence = (
        output_sequences_data.genres_ids_sequence.apply(
            lambda ids: create_sequences(ids, sequence_length, step_size)
        )
    )

    output_sequences_data.rating_sequence = output_sequences_data.rating_sequence.apply(
        lambda ids: create_sequences(ids, sequence_length, step_size)
    )

    return output_sequences_data


def extract_release_year(title):
    pattern = r"\((\d{4})\)"

    match = re.search(pattern, title)
    year = match.group(1)
    return int(year)


def get_origin_title(title):
    pattern = r"\s*\(\d{4}\)"
    return re.sub(pattern, "", title)
