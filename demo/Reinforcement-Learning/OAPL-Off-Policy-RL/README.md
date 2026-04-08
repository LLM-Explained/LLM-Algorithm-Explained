# OAPL Off-Policy RL Backbone

This repository provides a **core backbone** for the main algorithmic idea behind **OAPL**:

LLM RL post-training is often off-policy by design, so the update rule should explicitly account for mismatch between the lagged inference policy and the current training policy.

## What this code covers

This scaffold includes the major algorithmic pieces:

- a **lagged inference policy**
- rollout collection from that lagged policy
- a **trainable policy** updated from off-policy data
- a **KL-style anchor** that keeps the trainable policy close to the rollout policy

## Core algorithm

The backbone implemented here is:

1. collect rollouts from a lagged inference policy
2. estimate advantages from those off-policy samples
3. optimize the training policy on those samples
4. penalize excessive drift from the lagged inference policy with a KL term

In compact form:

```text
maximize   E_offpolicy[ A(y) * log pi_train(y) ]  -  tau * KL(pi_train || pi_infer_lagged)
```
