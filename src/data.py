from collections import namedtuple
from typing import Optional, Tuple

from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T


DataInfo = namedtuple('DataInfo', 'image_channels image_size sigma_data')
DataLoaders = namedtuple('DataLoaders', 'train valid')


class CharDataset(Dataset):
    """
    Emits batches of characters.

    Adapted from "https://github.com/karpathy/minGPT".
    """
    def __init__(self, config, data):

        chars = ... # get characters from the input data
        self.stoi = { ch:i for i,ch in enumerate(chars) } # map characters to integer indices
        pass

    def get_vocab_size(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        # encode every character to an integer
        # return the chunk and the shifted version as tensors
        pass

def load_dataset_and_make_dataloaders(
        dataset_name: str,
        root_dir: str,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False    
    ) -> Tuple[DataLoaders, DataInfo]:

    train_dataset, valid_dataset, data_info = load_dataset(dataset_name, root_dir)
    dl = make_dataloaders(train_dataset, valid_dataset, data_info.num_classes, batch_size, num_workers, pin_memory)
    return dl, data_info


def load_dataset(dataset_name='Shakespeare', root_dir='data') -> Tuple[Dataset, Dataset, DataInfo]:

    match dataset_name:
        
        case 'Shakespeare':
            t = T.Compose([T.ToTensor(), T.Pad(2), T.Normalize(mean=(0.5,), std=(0.5,))])
            train_dataset = FashionMNIST(root_dir, download=True, transform=t)
            train_dataset, valid_dataset = random_split(train_dataset, [50000, 10000])  # both come from the training set
            num_classes = 10
        case other:
            raise RuntimeError('Unknown dataset: ' + other)

    x, _ = next(iter(DataLoader(train_dataset, batch_size=10000, shuffle=True)))
    _, c, h, w = x.size()
    assert h == w
    sigma_data = x.std()
    
    return train_dataset, valid_dataset, DataInfo(c, h, sigma_data)


def make_dataloaders(
        train_dataset: Dataset,
        valid_dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False    
    ) -> DataLoaders:
    
    kwargs = {'num_workers': num_workers, 'persistent_workers': (num_workers > 0), 'pin_memory': pin_memory}
    
    return DataLoaders(
        train=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs),
        valid=DataLoader(valid_dataset, batch_size=2 * batch_size, **kwargs)  
    )
