import torch.nn as nn
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def __getitem__(self, index):
        pass
