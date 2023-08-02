import torch
from torch.utils.data import Dataset


class RatingDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item_dict = self.data.iloc[index].to_dict()

        dtype_dict = {}
        for k, v in item_dict.items():
            dtype_dict[k] = torch.long
        dtype_dict["target_rating"] = torch.float32
        dtype_dict["sex"] = torch.float32

        sample = {}
        for k, v in item_dict.items():
            sample[k] = torch.tensor(v, dtype=dtype_dict[k]).to(self.device)

        return sample
