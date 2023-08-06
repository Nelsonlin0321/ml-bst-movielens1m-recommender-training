# # A Behavior Sequence Transformer For Next Movie Recommendation
#
# **Author:** [Nelson Lin](https://www.linkedin.com/in/nelson-lin-842564164/)<br>
# **Description:** Rating rate prediction using the Behavior Sequence Transformer (BST) model on the Movielens 1M.

import os
import dotenv
import torch
from torch import nn
import numpy as np
import random
import mlflow
from uuid import uuid4
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import DataLoader
from src import utils
from src.utils import Config
from src.dataset import RatingDataset
from src.model import BSTRecommenderModel
from src.eval import evaluate
from prepare_data import DataPreparer
from prefect import flow, task, get_run_logger
import argparse

# import logging
# logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
#                     datefmt='%Y-%m-%d:%H:%M:%S',
#                     level=logging.INFO)

# logger = logging.getLogger(__name__)

logger = get_run_logger()


dotenv.load_dotenv("./.env")

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

TRACKING_URL = os.getenv("TRACKING_URL", "http://175.41.182.223:5050/")
EXPERIMENT_NAME = os.getenv(
    "EXPERIMENT_NAME", "bst-movielens1m-recommender-training")

parser = argparse.ArgumentParser(
    description='Training BST model for movie recommendation')

parser.add_argument('--artifact_dir', type=str, required=False,
                    default="artifacts", help="the directory to save artifacts")
parser.add_argument('--model_save_dir', type=str, required=False,
                    default="models", help="the directory to save models")
parser.add_argument('--sequence_length', type=int, required=False,
                    default=6, help="the length of sequence to genreate")
parser.add_argument('--test_size', type=float, required=False,
                    default=0.85, help="percentage of test data to allocate")
parser.add_argument('--genres_length', type=int, required=False,
                    default=4, help="the length of genres sequence to pad")
parser.add_argument('--embedding_dim', type=int, required=False,
                    default=32, help="The number of dimension for embedding")
parser.add_argument('--dropout', type=float, required=False,
                    default=0.2, help="dropout")
parser.add_argument('--epoches', type=int, required=False,
                    default=5, help="Number of epoch for training")
parser.add_argument('--learning_rate', type=float, required=False,
                    default=0.001, help="Learning Rate")
parser.add_argument('--batch_size', type=int, required=False,
                    default=128, help="Batch Size")
parser.add_argument('--env', type=str, required=False,
                    default="dev", help="test,dev and prod for env setting")


def get_config(data_preparer: DataPreparer, args) -> Config:

    logger.info("Getting model config from data preparation")
    num_movie = len(data_preparer.movie_id_map_dict)
    num_age_group = len(data_preparer.age_group_id_map_dict)
    num_genre = len(data_preparer.genres_map_dict)

    embed_configs = {}
    EMED_DIM = args.embedding_dim
    embed_configs["movie"] = {"embed_dim": EMED_DIM, "num_embed": num_movie}
    embed_configs["genre"] = {"embed_dim": EMED_DIM, "num_embed": num_genre}
    embed_configs["age_group"] = {
        "embed_dim": EMED_DIM, "num_embed": num_age_group}
    embed_configs["position"] = {
        "embed_dim": EMED_DIM, "num_embed": data_preparer.sequence_length}
    config_dict = {}
    config_dict["embed_configs"] = embed_configs
    config_dict["transformer_num_layer"] = 3
    config_dict["dropout"] = args.dropout
    config_dict["epoches"] = args.epoches
    config_dict["learning_rate"] = args.learning_rate
    config_dict["batch_size"] = args.batch_size
    config_dict["sequence_length"] = data_preparer.sequence_length
    config_dict["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config_dict["env"] = args.env

    utils.save_json(config_dict, os.path.join(
        args.artifact_dir, "config.json"))

    config = utils.Config(dict=config_dict)

    return config


class Trainer():
    def __init__(self, config, min_max_scaler, train_data, test_data, artifact_dir, model_save_dir) -> None:
        self.config = config
        self.min_max_scaler = min_max_scaler
        self.train_data = train_data
        self.test_data = test_data
        self.artifact_dir = artifact_dir
        self.model_save_dir = model_save_dir
        self.train_dataset = RatingDataset(data=train_data)
        self.test_dataset = RatingDataset(data=test_data)
        self.batch_size = config.batch_size
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=True)

        self.model = BSTRecommenderModel(config).to(config.device)

        self.loss_func = nn.L1Loss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate)
        self.total_batch = 0
        self.best_eval_loss = float("inf")
        self.best_checkpoint = 0
        self.model_version = str(uuid4())
        self.config.eval_steps = len(self.train_loader) // 3
        self.base_step = 64

    def train(self) -> str:
        """ The entry point of training

        Returns:
            _type_:str:The path of the best model
        """
        before_training_metrics = evaluate(
            self.model, self.test_loader, self.min_max_scaler, self.loss_func)
        logger.info(f"The metrics before training:{before_training_metrics}")

        mlflow.set_tracking_uri(TRACKING_URL)
        logger.info(f"Mlflow tracking URL is {TRACKING_URL}")
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"Mlflow experiment name is {EXPERIMENT_NAME}")
        total_pbar = tqdm(total=len(self.train_loader) * self.config.epoches,
                          desc="Training", position=0, leave=True)

        metrics_list = []
        best_model_path = None
        with mlflow.start_run():
            mlflow.set_tag("ENV", self.config.env)
            mlflow.log_params(self.config.dict)
            mlflow.log_artifact("./src", artifact_path="")
            mlflow.log_artifact(self.artifact_dir, artifact_path="")

            for epoch in range(self.config.epoches):
                train_loss_list = []
                prob_list = []
                rating_list = []

                for inputs in self.train_loader:
                    model = self.model.train()
                    self.optimizer.zero_grad()
                    probs = model(inputs)

                    rating = inputs["target_rating"].view(-1, 1)

                    loss = self.loss_func(probs, rating)
                    loss.backward()
                    self.optimizer.step()
                    train_loss_list.append(loss.item())

                    probs = probs.detach().cpu().numpy().flatten().tolist()
                    prob_list.extend(probs)
                    rating = rating.detach().cpu().flatten().tolist()
                    rating_list.extend(rating)

                    if (self.total_batch + 1) % self.config.eval_steps == 0:
                        current_steps = ((self.total_batch + 1) *
                                         self.batch_size) // self.base_step
                        improve = False
                        model_metrics = evaluate(
                            model, self.test_loader, self.min_max_scaler, self.loss_func)
                        eval_loss = model_metrics["eval_loss"]

                        if eval_loss <= self.best_eval_loss:
                            improve = True
                            best_checkpoint = current_steps
                            self.best_eval_loss = eval_loss

                        train_loss = np.mean(train_loss_list)

                        real_ratings = self.min_max_scaler.inverse_transform(
                            np.array(rating_list).reshape(-1, 1)
                        )[:, 0]
                        prediction_ratings = self.min_max_scaler.inverse_transform(
                            np.array(prob_list).reshape(-1, 1)
                        )[:, 0]
                        MAE = metrics.mean_absolute_error(
                            real_ratings, prediction_ratings)
                        MSE = metrics.mean_squared_error(
                            real_ratings, prediction_ratings)

                        model_metrics["best_eval_loss"] = self.best_eval_loss
                        model_metrics["train_loss"] = train_loss
                        model_metrics["train_MAE"] = MAE
                        model_metrics["train_MSE"] = MSE

                        model_metrics["steps"] = current_steps
                        model_metrics["best_checkpoint"] = best_checkpoint
                        metrics_list.append(model_metrics)

                        for metrics_name, metrics_value in model_metrics.items():
                            mlflow.log_metric(metrics_name, metrics_value)

                        if improve:
                            save_dir = os.path.join(
                                self.model_save_dir, self.model_version)
                            os.makedirs(save_dir, exist_ok=True)
                            model_path = utils.save_model(
                                model, save_dir, current_steps, model_metrics)
                            best_model_path = model_path
                            mlflow.log_artifact(
                                model_path, artifact_path="model")
                            mlflow.pytorch.log_model(model, "model")

                        post_fix_message = {k: round(v, 3)
                                            for k, v in model_metrics.items()}
                        total_pbar.set_postfix(post_fix_message)

                        model = model.train()

                    self.total_batch += 1
                    total_pbar.update(1)

                model = model.train()

            total_pbar.close()

        best_model_path = os.path.abspath(best_model_path)
        logger.info(f"The best model is save at local :{best_model_path} ")
        return best_model_path


@task(log_prints=True)
def train(args, data_preparer):
    model_config = get_config(data_preparer, args)

    train_data = data_preparer.train_data
    test_data = data_preparer.test_data

    env = args.env

    if str(env) == 'test':
        sample_size_for_testing = 10000
        train_data = train_data.head(sample_size_for_testing)
        test_data = test_data.head(sample_size_for_testing)
        model_config.epoches = 2

    trainer = Trainer(config=model_config, min_max_scaler=data_preparer.min_max_scaler,
                      train_data=train_data, test_data=test_data,
                      artifact_dir=args.artifact_dir, model_save_dir=args.model_save_dir)

    trainer.train()


@task(log_prints=True)
def prepare_data(args):
    artifact_dir = args.artifact_dir
    sequence_length = args.sequence_length
    val_size = args.test_size
    genres_length = args.genres_length
    data_preparer = DataPreparer(artifact_dir=artifact_dir, sequence_length=sequence_length,
                                 test_size=val_size, genres_length=genres_length)

    data_preparer.prepare_data()
    return data_preparer


@flow(log_prints=True)
def bst_movielens1m_recommender_training_pipeline(env=None, dropout=None,
                                                  epoches=None, test_size=None,
                                                  batch_size=None, artifact_dir=None,
                                                  embedding_dim=None, genres_length=None,
                                                  learning_rate=None, model_save_dir=None,
                                                  sequence_length=None, **kwargs):

    args = parser.parse_args()

    if env is not None:
        args.env = env

    print(f"Execution Env: {args.env}")

    if dropout is not None:
        args.dropout = dropout

    if epoches is not None:
        args.epoches = epoches

    if test_size is not None:
        args.test_size = test_size

    if batch_size is not None:
        args.batch_size = batch_size

    if artifact_dir is not None:
        args.artifact_dir = artifact_dir

    if genres_length is not None:
        args.env = genres_length

    if embedding_dim is not None:
        args.env = embedding_dim

    if embedding_dim is not None:
        args.env = embedding_dim

    if learning_rate is not None:
        args.learning_rate = learning_rate

    if model_save_dir is not None:
        args.model_save_dir = model_save_dir

    if sequence_length is not None:
        args.sequence_length = sequence_length

    assert str(args.env) in ['test', 'dev', 'prod']

    data_preparer = prepare_data(args=args)
    train(args=args, data_preparer=data_preparer)


if __name__ == "__main__":
    bst_movielens1m_recommender_training_pipeline()
