import hydra
import torch

from init import init_sampling
from data import DataLoaders

from omegaconf import DictConfig


@torch.inference_mode()
def sample(
        model: torch.nn.Module,
        dl: DataLoaders,
        device: torch.device|str,
        context: str,  # TODO: multiple prompts, would require a padding token
        nb_tokens: int=50,
        sampling_mode: str="prob",
        temperature: float=1
    ) -> torch.Tensor:
    tokenize = lambda ctx: torch.tensor(list(map(lambda s: dl.train.dataset.stoi[s], ctx)), device=device)
    tokens_to_string = lambda tokens: "".join(list(map(lambda i: dl.train.dataset.itos[i.item()], tokens)))
    model.eval()

    tokenized_context = tokenize(context)
    y = model.generate(tokenized_context[None, :], nb_tokens=nb_tokens, sampling_mode=sampling_mode, temperature=temperature)
    completions = [tokens_to_string(tokens) for tokens in y]

    model.train()
    return completions

@torch.inference_mode()
@hydra.main(version_base=None, config_path="../config", config_name="config")
def sampler(cfg: DictConfig):
    model, dl, device, sampling_mode, path, temperature_str = init_sampling(cfg)
    # Sample
    completions = sample(
                    model, dl, device,
                    context=cfg.common.sampling.context,
                    nb_tokens=cfg.common.sampling.nb_tokens,
                    sampling_mode=sampling_mode,
                    temperature=cfg.common.sampling.temperature
                )
    with open(path / f"completions_temp_{temperature_str}.txt", "a") as f:
        for completion in completions:
            print(cfg.common.sampling.context + completion, file=f)

if __name__ == "__main__":
    sampler()