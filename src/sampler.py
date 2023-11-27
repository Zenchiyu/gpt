import hydra
import torch

from init import init

from omegaconf import DictConfig
from typing import Optional, Callable


@hydra.main(version_base=None, config_path="../config", config_name="config")
def sampler(cfg: DictConfig):
    # Initialization
    init_tuple = init(cfg)
    model, info = init_tuple.model, init_tuple.info

    # Don't use the checkpoint seed for sampling
    torch.seed()

    # Sample and display
    # TODO: arguments needed for sampling
    # TODO: sample here
    # TODO: save sampled txt

if __name__ == "__main__":
    sampler()