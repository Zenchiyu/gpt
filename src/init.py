import torch
import torch.nn as nn
import torch.optim as optim

from models.transformer import Transformer
from data import load_dataset_and_make_dataloaders

from collections import namedtuple
from datetime import date
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Any


Init = namedtuple('Init', 'model optimizer criterion '+\
                  'dl device nb_steps_finished '+\
                  'begin_date save_path chkpt_path')
InitSample = namedtuple('InitSample', 'model dl device '+\
                        'sampling_mode path temperature_str')

def create_save_directories(cfg: DictConfig) -> tuple[Path, Path]:
    """
    Create directories for saving samples and checkpoints.
    """
    save_path, chkpt_path = Path(cfg.common.sampling.save_path), Path(cfg.common.training.chkpt_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    chkpt_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path, chkpt_path

def load_chkpt(chkpt_path: Path, device: str|torch.device) -> tuple[Any, int, str]:
    """
    Load checkpoint if exists, set random seed, and handle run resuming.
    
    Note: seed is used to get the same training, validation sets splits
    when resuming our runs.

    Credits: https://fleuret.org/dlc/materials/dlc-handout-11-4-persistence.pdf
    """
    chkpt, nb_steps_finished, begin_date, seed = None, 0, str(date.today()), torch.initial_seed()  # by default: random seed
    try:
        chkpt = torch.load(chkpt_path, map_location=device)
        nb_steps_finished = chkpt.get("nb_steps_finished", nb_steps_finished)
        begin_date = chkpt.get("begin_date", begin_date)
        seed = chkpt.get("seed", seed)
        torch.manual_seed(seed)
        print(f"\nStarting from checkpoint with {nb_steps_finished} finished steps"+\
              f", and initial seed {seed} (=> same datasets).")
    except FileNotFoundError:
        print(f"Starting from scratch with random initial seed {seed}.")

    return chkpt, nb_steps_finished, begin_date

def init(cfg: DictConfig, verbose: bool=True) -> Init:
    if verbose:
        print("Config:")
        print(OmegaConf.to_yaml(cfg))

    gpu    = torch.cuda.is_available()
    device = torch.device('cuda:0' if gpu else 'cpu')
    
    ## Create save & chkpt directories
    save_path, chkpt_path = create_save_directories(cfg)

    ## Load checkpoint if exists
    chkpt, nb_steps_finished, begin_date = load_chkpt(chkpt_path, device)

    ## DataLoaders
    dl = load_dataset_and_make_dataloaders(
        dataset_path=cfg.dataset.path,          # where the dataset is stored as .txt file
        chunk_size=cfg.dataset.chunk_size,      # e.g. 128 (= max seq length)
        batch_size=cfg.dataset.batch_size,      # e.g. 128
        num_workers=cfg.dataset.num_workers,    # can use more workers if GPU is waiting for the batches
        pin_memory=gpu,                         # use pin memory if plan to move the data to GPU
    )
    
    ## Model and criterion
    model = Transformer(
        vocab_size=dl.train.dataset.get_vocab_size(),
        max_seq_len=cfg.model.max_seq_len,      # = chunk size
        embed_dim=cfg.model.embed_dim,
        mlp_hidden_dim=cfg.model.mlp_hidden_dim,
        nb_layers=cfg.model.nb_layers,
        nb_heads=cfg.model.nb_heads
    )
    criterion = nn.CrossEntropyLoss()
    model.to(device=device)
    criterion.to(device=device)

    ## Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr)  # TODO: learning rate schedule

    ## Load saved model and optimizer state dict if chkpt exists
    if chkpt:
        model.load_state_dict(chkpt["model_state_dict"])
        optimizer.load_state_dict(chkpt["optimizer_state_dict"])
        print("\nSuccessfully loaded model & optimizer state dicts.")

    print(f"\n\nDataset: {Path(cfg.dataset.path).stem}, Using device: {device}")

    return Init(model, optimizer, criterion,
                dl, device, nb_steps_finished,
                begin_date, save_path, chkpt_path)

# TODO: fix this
def init_sampling(cfg: DictConfig) -> InitSample:
    seed = torch.random.initial_seed()  # retrieve current seed

    # Initialization
    init_tuple = init(cfg)  # TODO: make it more efficient
    model, dl, device = init_tuple.model, init_tuple.dl, init_tuple.device
    del init_tuple
    try:
        sampling_mode = cfg.common.sampling.sampling_mode
    except:
        sampling_mode = "prob"
    dataset_name = str.lower(Path(cfg.dataset.path).stem)
    temperature_str = str(cfg.common.sampling.temperature).replace('.','_')
    path = Path(f"./results/txts/{dataset_name}/{sampling_mode}/")
    path.mkdir(parents=True, exist_ok=True)

    # Don't use the checkpoint seed for sampling
    torch.manual_seed(seed)
    return InitSample(model, dl, device, sampling_mode, path, temperature_str)