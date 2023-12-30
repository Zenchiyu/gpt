# Small Language Model
Deep Learning Project to generate Shakespearean-like text using a transformer-based, character-level language model (GPT-like).

## Installation

- Install PyTorch

```
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

- And other dependencies:
```
pip3 install -r requirements.txt
```

## Training

If you want to train a model from scratch, please ensure the checkpoint path doesn't point to an existing checkpoint. Otherwise, training will resume.
```bash
python src/trainer.py
```

## Sampling
```bash
python src/sampler.py
```
