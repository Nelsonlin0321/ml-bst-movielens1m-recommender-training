from sklearn import metrics
from tqdm import tqdm
import torch
import numpy as np
from torch import nn


def evaluate(model, dataset_loader, min_max_scaler, loss_func=nn.L1Loss()):
    model.eval()

    prob_list = []
    rating_list = []
    eval_loss_list = []

    # loss_func = nn.MSELoss()

    pbar = tqdm(total=len(dataset_loader),
                desc="Evaluating", position=0, leave=True)

    for inputs in dataset_loader:
        with torch.no_grad():
            probs = model(inputs)
            ratings = inputs["target_rating"].view(-1, 1)

            loss = loss_func(probs, ratings)
            eval_loss_list.append(loss.item())

            probs = probs.cpu().numpy().flatten().tolist()
            prob_list.extend(probs)

            ratings = ratings.cpu().numpy().flatten().tolist()
            rating_list.extend(ratings)

            pbar.update(1)

    pbar.close()

    real_ratings = min_max_scaler.inverse_transform(
        np.array(rating_list).reshape(-1, 1)
    )[:, 0]
    prediction_ratings = min_max_scaler.inverse_transform(
        np.array(prob_list).reshape(-1, 1)
    )[:, 0]

    MAE = metrics.mean_absolute_error(real_ratings, prediction_ratings)
    MSE = metrics.mean_squared_error(real_ratings, prediction_ratings)

    eval_metrics = {}
    eval_metrics["eval_loss"] = sum(eval_loss_list) / len(eval_loss_list)
    eval_metrics["eval_MAE"] = MAE
    eval_metrics["eval_MSE"] = MSE

    return eval_metrics
