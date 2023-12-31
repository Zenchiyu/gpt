# Small Language Model
Deep Learning Project to generate Shakespearean-like text using a transformer-based, character-level language model (GPT-like).

Generating $1000$ characters using `O God, O God!` as prompt:
```
O God, O God! she is full of death.
Patience, I will take thy hire with thy death.

LADY GREY:
Why, then I will do what your grace commands.

GLOUCESTER:

CLARENCE:

LADY GREY:
Why stops my lord; and so much before
The mother of our gates to die.

KING EDWARD IV:
Thanks, good Montgomery; and my soul
Did triumph with all duteous pleasing well
Will not procuous here, stults, and as ours,
Stand all your worthy answer the law,
And afterward here with her! Were I a tyrant,
Where were her life? she durst not call me so,
If she did know me one. Away with him to prison!
He would not prepare to fine his country
And by his power increaseth himself as well as
In all proof, as it appears, cracking ten times,
Not am body's parent of pardon.

MISTRESS OVERDONE:
Nay, but I know 'tis so: I saw him arrested, saw
spectatord and as it were far the people,
To lose his hum with his fower a fool.

SICINIUS:
He would not like the dews of a week fellow;
hath bid he has march him well; which was it born.
To see how the pow
```

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
