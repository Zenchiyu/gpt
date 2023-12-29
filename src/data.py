import numpy as np
import torch

from collections import namedtuple
from typing import Optional, Tuple
from torch.utils.data import DataLoader, Dataset, random_split

from pathlib import Path


DataLoaders = namedtuple('DataLoaders', 'train valid test')

class CharDataset(Dataset):
    """
    Emits batches of characters.

    Adapted from "https://github.com/karpathy/minGPT".
    """
    def __init__(self,
                 chunk_size: int,
                 data: str,
                 split: str="train",
                 train_ratio: float=90/100
            ) -> None:
        # TODO: random split, clean code
        idxs = np.cumsum([0, int(train_ratio*len(data)), int((1-train_ratio)*len(data)/2), np.ceil((1-train_ratio)*len(data)/2)], dtype=int)
        self.data = dict(zip(["train", "val", "test"], [data[b:e] for b, e in zip(idxs[:-1], idxs[1:])]))
        
        chars = list(dict.fromkeys(self.data["train"]))  # get characters from the input data

        self.chunk_size = chunk_size
        # TODO: special tokens like pad token!
        self.stoi = {ch:i for i,ch in enumerate(chars)}  # map characters to integer indices
        self.itos = {i:ch for ch, i in self.stoi.items()}
        self.ids = torch.tensor(list(map(lambda s: self.stoi[s], self.data[split])))

    def get_vocab_size(self):
        return len(self.stoi)

    def __len__(self):
        return len(self.ids)-self.chunk_size

    def __getitem__(self, idx):
        # grab a chunk of chunk_size characters from the data
        # encode every character to an integer
        # return the chunk and the shifted version as tensors

        # DataLoader will take care of the constant chunk_size 
        return self.ids[idx:idx+self.chunk_size], self.ids[idx+1:idx+self.chunk_size+1]


def load_dataset_and_make_dataloaders(
        dataset_name: str,
        root_dir: str,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False    
    ) -> DataLoaders:

    train_dataset, valid_dataset, test_dataset = load_dataset(dataset_name, root_dir)
    dl = make_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size, num_workers, pin_memory)
    return dl

def load_dataset(dataset_name='Shakespeare', root_dir='data') -> Tuple[Dataset, Dataset]:
    with open(Path(root_dir) / f"{dataset_name}.txt", "r") as f:
        data = "".join(f.readlines())
    
    train_dataset = CharDataset(10, data, split="train")
    valid_dataset = CharDataset(10, data, split="val")
    test_dataset = CharDataset(10, data, split="test")
    
    return train_dataset, valid_dataset, test_dataset


def make_dataloaders(
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False    
    ) -> DataLoaders:

    kwargs = {'num_workers': num_workers, 'persistent_workers': (num_workers > 0), 'pin_memory': pin_memory}
    
    return DataLoaders(
        train=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs),
        valid=DataLoader(valid_dataset, batch_size=2 * batch_size, **kwargs),
        test=DataLoader(test_dataset, batch_size=2 * batch_size, **kwargs)
    )