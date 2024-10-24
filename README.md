# Get Started

The Python version used is 3.10.6.

This repository contains the code from our efforts to formalize a scalable theory of mind in multi-agent systems. We published the design motivation for the architecture used here in a preliminary work at the Agentic Markets workshop at ICML 2024, titled [Scalable Approaches for a Theory of Many Minds](https://openreview.net/forum?id=P0oG5gDh6T).

## Clone the repository

```bash
git clone git@github.com:amimem/ToMM.git
cd ToMM
```

## Install requirements

Create a virtual environment and install the requirements.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the code

```bash
python mlp_match.py ARGUMENTS
```

If you are using Slurm, create a symlink to the scratch folder for the logs.

```bash
ln -s $SCRATCH scratch
sbatch scripts/slurm.sh
```
