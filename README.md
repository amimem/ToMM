## Install
Use micromamba
```bash
micromamba create -n tomm -f env.yaml 
micromamba activate tomm
```

To update `env.yaml`:
```bash
micromamba env export --from-history > env.yaml
```

# Experiments
## 20241127-tomnet (current commit)
### Reproducibility and results
Ran `python bandit_tomnet.py` (hard-coded args, use commit for repro.).  
320 runs here: https://wandb.ai/abstraction/ToMMM/workspace?nw=9xprbzmsc5i

### Notes
This is just a ToMNet baseline for bandit settings, using a student-faculty (i.e. student - many teachers) setup:

**Faculty network:**
- Sample `bsz` random continuous state vectors of size `dim_states` 
- For each teacher:
  - Using teacher-specific `observation_fn` MLPs, map the state to an observation vector of size `dim_observation`.
  - Using one shared `policy_fn` MLP, map observations to action logits of size `num_actions` and argmax to obtain gold-standard actions.
- Repeat this process, sampling `history_len` state vectors to get a context of past state/action pairs for each teacher.

**Student network**: (Bandit ToMNet, Based on [Machine Theory of Mind](https://arxiv.org/pdf/1802.07740) A.3.2 implementation)

- For each teacher:
  -  Map past state/action pairs to character embedding using a shared `CharNet` encoder.
  - Concatenate current state embedding with character embedding and feed into a shared `PredictionNet` decoder to obtain action logits.
  - Compute `cross_entropy` loss using gold-standard actions.
