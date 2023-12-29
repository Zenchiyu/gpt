import numpy as np
import torch

from collections import namedtuple
from typing import Optional, Tuple
from torch.utils.data import DataLoader, Dataset, random_split

from pathlib import Path


DataLoaders = namedtuple('DataLoaders', 'train valid test')

class CharDataset(Dataset):
    """
    Emits a sequence of token ids and its shifted version.
    """
    def __init__(self,
                 data: str,
                 chunk_size: int,
                 split: str="train",
                 train_ratio: float=90/100
            ) -> None:
        # TODO: random split, clean code
        idxs = np.cumsum([0, int(train_ratio*len(data)), int((1-train_ratio)*len(data)/2), np.ceil((1-train_ratio)*len(data)/2)], dtype=int)
        self.data = dict(zip(DataLoaders._fields, [data[b:e] for b, e in zip(idxs[:-1], idxs[1:])]))
        
        chars = list(dict.fromkeys(self.data["train"]))  # get characters from the input data

        self.chunk_size = chunk_size
        # TODO: special tokens like pad token!
        self.stoi = {ch:i for i,ch in enumerate(chars)}  # map characters to integer indices
        self.itos = {i:ch for ch, i in self.stoi.items()}
        self.ids = torch.tensor(list(map(lambda s: self.stoi[s], self.data[split])))

    def get_vocab_size(self) -> int:
        return len(self.stoi)

    def __len__(self) -> int:
        return len(self.ids)-self.chunk_size

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # grab a chunk of chunk_size characters from the data
        # encode every character to an integer
        # return the chunk and the shifted version as tensors

        # DataLoader will take care of the constant chunk_size 
        return self.ids[idx:idx+self.chunk_size], self.ids[idx+1:idx+self.chunk_size+1]


def load_dataset_and_make_dataloaders(
        dataset_path: str=str(Path("data/shakespeare.txt")),
        chunk_size: int=128,
        batch_size: int=128,
        num_workers: int = 0,
        pin_memory: bool = False    
    ) -> DataLoaders:

    train_dataset, valid_dataset, test_dataset = load_dataset(dataset_path=dataset_path, chunk_size=chunk_size)
    dl = make_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size, num_workers, pin_memory)
    return dl

def load_dataset(dataset_path: str, chunk_size: int) -> Tuple[Dataset, ...]:
    with open(dataset_path, "r") as f:
        data = "".join(f.readlines())
    
    train_dataset, valid_dataset, test_dataset = [CharDataset(data, chunk_size, split=split) for split in DataLoaders._fields]
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