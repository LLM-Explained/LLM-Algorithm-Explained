# Reasoning Judge RL Backbone

This repository provides a **core backbone** for the main algorithmic idea behind recent work on **reasoning LLMs-as-judges in non-verifiable RL post-training**:

a judge is not just an evaluator — once it sits inside the RL loop, it becomes the reward model that the policy is optimized against.

## What this code covers

This scaffold includes the major algorithmic pieces:

- a **policy** that proposes responses
- a **judge** that assigns rewards
- a **policy update rule** that increases judged reward
- two judge types:
  - a **non-reasoning judge**
  - a **reasoning judge**
- a **gold evaluator** used only for analysis

## Core algorithm

The backbone implemented here is:

1. sample responses from the policy
2. score them with the judge
3. update the policy to increase expected judged reward
4. compare how the policy evolves under different judges

In compact form:

```text
maximize   E_{y ~ pi}[ J(x, y) ]
```
