import pickle
import os
import torch
import json


def save_json(json_object, file_path):
    with open(file_path, mode='w', encoding='utf-8') as f:
        json.dump(json_object, f, indent=4, ensure_ascii=False)


def open_json(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f:
        json_object = json.load(f)
        return json_object


def open_object(object_path):
    with open(object_path, mode='rb') as f:
        obj = pickle.load(f)

    return obj


def save_object(object_path, obj):
    os.makedirs(os.path.dirname(object_path), exist_ok=True)
    with open(object_path, mode='wb') as f:
        pickle.dump(obj, f)


def save_model(model, model_save_dir, step, model_metrics):
    model_save_dir = os.path.join(model_save_dir, f"checkpoint-{step}")
    model_name = "pytorch_model.pt"
    train_state_name = "training_state.json"
    os.makedirs(model_save_dir, exist_ok=True)

    model_path = os.path.join(model_save_dir, model_name)
    train_state_path = os.path.join(model_save_dir, train_state_name)

    torch.save(model.state_dict(), model_path)

    if model_metrics is not None:
        save_json(model_metrics, train_state_path)

    return model_path


class Config:
    def __init__(self, dict):
        self.dict = dict
        for key, value in dict.items():
            setattr(self, key, value)
