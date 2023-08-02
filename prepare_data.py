# # A Behavior Sequence Transformer For Next Movie Recommendation
#
# **Author:** [Nelson Lin](https://www.linkedin.com/in/nelson-lin-842564164/)<br>
# **Description:** Rating rate prediction using the Behavior Sequence Transformer (BST) model on the Movielens 1M.
import pandas as pd
import numpy as np
from src import utils
from src import data_utils
from typing import List
import argparse
import logging
import os
import random


logging.basicConfig(format='%(asctime)s,%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description='Data preparation for downstream training')

parser.add_argument('--artifact_dir', type=str, required=False,
                    default="artifacts", help="the directory to save artifacts")
parser.add_argument('--sequence_length', type=int, required=False,
                    default=6, help="the length of sequence to genreate")
parser.add_argument('--test_size', type=float, required=False,
                    default=0.85, help="percentage of test data to allocate")
parser.add_argument('--genres_length', type=int, required=False,
                    default=4, help="the length of genres sequence to pad")


class DataPreparer():
    def __init__(self, artifact_dir, sequence_length, test_size, genres_length) -> None:
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.genres_length = genres_length
        logger.info(
            "Downloading data from http://files.grouplens.org/datasets/movielens/ml-1m.zip")
        data_utils.download_data()
        self.users, self.ratings, self.movies = data_utils.read_source_data()

        self.train_data = None
        self.test_data = None

    def save_movie_info(self):

        movies_info = self.movies.copy()
        movies_info['release_year'] = movies_info['title'].apply(
            lambda x: data_utils.extract_release_year(x))

        movies_info['genres'] = movies_info['genres'].apply(
            lambda x: x.split("|"))

        movies_info['origin_title'] = movies_info['title'].apply(
            lambda x: data_utils.get_origin_title(x))

        movies_info.to_parquet(os.path.join(
            self.artifact_dir, "movie_info.parquet"))

    def encode_features(self) -> None:
        """ To encode the features 

        Args:
            movies (pd.DataFrame): _description_
            ratings (pd.DataFrame): _description_
            users (pd.DataFrame): _description_

        Returns:
            Tuple[pd.DataFrame]: _description_
        """

        self.sex_id_map_dict = data_utils.encode_sex(users=self.users)
        utils.save_object(
            f"{self.artifact_dir}/sex_id_map_dict.pkl", self.sex_id_map_dict)

        """Age"""
        self.age_group_id_map_dict = data_utils.encode_id(
            self.users, col="age_group")
        utils.save_object(
            f"{self.artifact_dir}/age_group_id_map_dict.pkl", self.age_group_id_map_dict)

        """Rating"""
        self.min_max_scaler = data_utils.encode_rating(self.ratings)
        utils.save_object(
            f"{self.artifact_dir}/rating_min_max_scaler.pkl", self.min_max_scaler)

        """Movie"""
        self.movie_id_map_dict = data_utils.encode_id(
            self.movies, col="movie_id")
        utils.save_object(
            f"{self.artifact_dir}/movie_id_map_dict.pkl", self.movie_id_map_dict)
        self.ratings["movie_id_index"] = self.ratings["movie_id"].map(
            self.movie_id_map_dict)

        """Genres"""
        self.genres_map_dict = data_utils.encode_genres(
            movies=self.movies, max_genres_length=self.genres_length)
        utils.save_object(
            f"{self.artifact_dir}/genres_map_dict.pkl", self.genres_map_dict)

        movies_to_genres_dict = self.movies[['movie_id_index', 'genres_ids']] \
            .set_index("movie_id_index")['genres_ids'].to_dict()
        utils.save_object(
            f"./{self.artifact_dir}/movies_to_genres_dict.pkl", movies_to_genres_dict)

    def transforme_to_sequence_data(self) -> pd.DataFrame:
        """ Gather Ratings, Movies and Users Dataframe to transform to be sequence 

        Args:
            ratings (pd.DataFrame): _description_
            movies (pd.DataFrame): _description_
            users (pd.DataFrame): _description_
            sequence_length (int, optional): _description_. Defaults to 5.

        Returns:
            pd.DataFrame: _description_
        """

        df_user_views = self.ratings[
            ["user_id", "movie_id_index", "norm_rating", "unix_timestamp"]
        ].merge(self.movies[["movie_id_index", "genres_ids"]], on=["movie_id_index"])

        df_agg = df_user_views.sort_values(
            by=["unix_timestamp"]).groupby("user_id")

        sequences = pd.DataFrame(
            data={
                "user_id": list(df_agg.groups.keys()),
                "movie_sequence": list(df_agg.movie_id_index.apply(list)),
                "genres_ids_sequence": list(df_agg.genres_ids.apply(list)),
                "rating_sequence": list(df_agg.norm_rating.apply(list)),
            }
        )

        sequence_lengths = range(2, self.sequence_length+1)

        df_list = [
            data_utils.generate_sequence_data(sequences, sequence_length)
            for sequence_length in sequence_lengths
        ]

        multi_sequence = pd.concat(df_list)

        multi_sequence_movies = multi_sequence[["user_id", "movie_sequence"]].explode(
            "movie_sequence", ignore_index=True
        )

        multi_sequence_rating = multi_sequence[["rating_sequence"]].explode(
            "rating_sequence", ignore_index=True
        )

        multi_sequence_genres = multi_sequence[["genres_ids_sequence"]].explode(
            "genres_ids_sequence", ignore_index=True
        )

        multi_sequence_transformed = pd.concat(
            [multi_sequence_movies, multi_sequence_rating,
                multi_sequence_genres], axis=1
        )

        multi_sequence_transformed = multi_sequence_transformed[
            multi_sequence_transformed["movie_sequence"].notnull()
        ]

        user_columns = ["user_id", "sex", "age_group_index"]

        multi_sequence_transformed = multi_sequence_transformed.merge(
            self.users[user_columns], on="user_id"
        )

        multi_sequence_transformed["sex"] = multi_sequence_transformed["sex"].astype(
            float)

        return multi_sequence_transformed

    def assign_rating(self, multi_sequence_transformed: pd.DataFrame) -> pd.DataFrame:
        """ Assign the last movie ratings as label

        Args:
            multi_sequence_transformed (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        multi_sequence_transformed["target_movie"] = multi_sequence_transformed[
            "movie_sequence"
        ].apply(lambda x: x[-1])
        multi_sequence_transformed["target_rating"] = multi_sequence_transformed[
            "rating_sequence"
        ].apply(lambda x: x[-1])

        # Assume that we don't have rating input from users in inference
        multi_sequence_transformed = multi_sequence_transformed.drop(
            "rating_sequence", axis=1)

        return multi_sequence_transformed

    def padding_genres_id(self, genres_ids_sequence) -> List[List[int]]:
        """ Padding genres id

        Args:
            genres_ids_sequence (_type_): _description_
            genres_map_dict (_type_): _description_
            max_sequence_length (_type_): _description_
            max_genres_length (_type_): _description_

        Returns:
            List[List[int]]: _description_
        """
        padding_list = [self.genres_map_dict["UNK"]] * self.genres_length
        for _ in range(self.sequence_length):
            genres_ids_sequence.append(padding_list)

        return genres_ids_sequence[:self.sequence_length]

    def padding_sequence(self, multi_sequence_transformed) -> pd.DataFrame:
        """Padding all features to be sequences

        Args:
            multi_sequence_transformed (pd.DataFrame): _description_
            movie_id_map_dict (Dict): _description_
            genres_map_dict (Dict): _description_
            max_genres_length (int, optional): _description_. Defaults to 4.

        Returns:
            pd.DataFrame: _description_
        """

        max_length = max([len(seq)
                          for seq in multi_sequence_transformed["movie_sequence"]])

        multi_sequence_transformed["movie_sequence"] = multi_sequence_transformed[
            "movie_sequence"
        ].apply(lambda x: x + max_length * [self.movie_id_map_dict["UNK"]])

        multi_sequence_transformed["movie_sequence"] = multi_sequence_transformed[
            "movie_sequence"
        ].apply(lambda x: x[:max_length])

        multi_sequence_transformed["genres_ids_sequence"] = multi_sequence_transformed[
            "genres_ids_sequence"
        ].apply(lambda x: self.padding_genres_id(x))

        return multi_sequence_transformed

    def train_test_split_and_save(self, multi_sequence_transformed: pd.DataFrame) -> None:
        """ Train Test Split and Save

        Args:
            multi_sequence_transformed (pd.DataFrame): _description_
            test_size (float, optional): _description_. Defaults to 0.85.
        """

        random_selection = np.random.rand(
            len(multi_sequence_transformed.index)) <= self.test_size
        train_data = multi_sequence_transformed[random_selection]
        test_data = multi_sequence_transformed[~random_selection]

        train_data = train_data.drop("user_id", axis=1)
        test_data = test_data.drop("user_id", axis=1)

        train_data.to_parquet(f"{self.artifact_dir}/train_data.parquet")
        test_data.to_parquet(f"{self.artifact_dir}/test_data.parquet")

        self.train_data = train_data
        self.test_data = test_data

    def prepare_data(self) -> None:

        logger.info("Save Movie Info")
        self.save_movie_info()

        logger.info("Encoding Features")
        self.encode_features()

        logger.info("Transforming to sequence")
        multi_sequence_transformed = self.transforme_to_sequence_data()

        logger.info("Assigning last ratings as labels")
        multi_sequence_transformed = self.assign_rating(
            multi_sequence_transformed)

        multi_sequence_transformed = self.padding_sequence(
            multi_sequence_transformed)

        logger.info("Train test split and save")
        self.train_test_split_and_save(multi_sequence_transformed)

        logger.info("Data preparation completed")


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    args = parser.parse_args()
    artifact_dir = args.artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)
    sequence_length = args.sequence_length
    test_size = args.test_size
    genres_length = args.genres_length

    data_preparer = DataPreparer(
        artifact_dir, sequence_length, test_size, genres_length)
    data_preparer.prepare_data()
