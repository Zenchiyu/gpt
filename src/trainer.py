import hydra
import torch
import wandb

from init import init
from utils import copy_config, copy_chkpt

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def train(cfg, run, dataset_seed,
          model, optimizer, criterion, dl, device, nb_steps_finished, begin_date, save_path, chkpt_path):
    step = nb_steps_finished
    losses = []
    while True:
        for X, Y in dl.train:
            X = X.to(device=device)  # N x L
            Y = Y.to(device=device)  # N x L
            
            logits = model(X)        # N x V x L
            loss = criterion(logits, Y)  # CE loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            losses.append(loss.item())

            if cfg.wandb.mode == "online":
                wandb.log({"step": step, "loss": losses[-1]})
            if step == cfg.common.nb_steps:
                return
            
            if step % 100 == 0:
                # Save checkpoint at each 100 finished steps
                torch.save({"nb_steps_finished": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": losses,
                            "seed": dataset_seed,
                            "begin_date": begin_date},
                            chkpt_path)
                if cfg.wandb.mode == "online":
                    copy_chkpt(run, begin_date, chkpt_path)
            

@hydra.main(version_base=None, config_path="../config", config_name="config")
def trainer(cfg: DictConfig):
    run_seed = torch.random.initial_seed()  # retrieve current seed in 
    # Initialization
    init_tuple = init(cfg)
    (model, optimizer, criterion, dl, device, nb_steps_finished, begin_date, save_path, chkpt_path) = init_tuple

    dataset_seed = torch.random.initial_seed()  # retrieve seed used for random split
    torch.manual_seed(run_seed)                 # reset seed
    nb_params = sum(map(lambda x: x.numel(), model.parameters()))
    print(f"\nModel: {model}\n\nNumber of parameters: {nb_params}")

    if cfg.wandb.mode == "online":
        run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                         **cfg.wandb)
        run.watch(model, criterion, log="all", log_graph=True)
        run.summary["nb_params"] = nb_params
        copy_config(run, begin_date=begin_date, config_name=HydraConfig.get()["job"]["config_name"])

    # Training
    train(cfg, run, dataset_seed, *init_tuple)
    
    if cfg.wandb.mode == "online":
        wandb.finish()

if __name__ == "__main__":
    trainer()