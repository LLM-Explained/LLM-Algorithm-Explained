# GRPO Backbone

This repository provides a **core backbone** for the main algorithmic idea behind **Group Relative Policy Optimization (GRPO)**:

replace critic-based advantage estimation with **group-relative normalized rewards** computed from multiple completions for the same prompt.

## What this code covers

This scaffold includes the major GRPO pieces:

- a policy that samples a **group** of completions for one prompt
- a reward function over those completions
- computation of **group-relative normalized advantages**
- a policy update driven by those relative advantages

## Core algorithm

For one prompt:

1. sample a group of completions
2. score each completion with a reward
3. compute:

```text
A_i = (r_i - mean(group_rewards)) / std(group_rewards)
```
