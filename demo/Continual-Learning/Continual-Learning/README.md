# Continual Learning Backbone Demo

This repository provides a **core backbone** for continual learning in sequential model adaptation.

It captures three major algorithmic ingredients that appear repeatedly in continual-learning systems for LLMs and foundation models:

- **sequential task stream**
- **replay buffer**
- **regularized updates against older parameters**

## What this code demonstrates

The demo compares three regimes:

1. **plain sequential fine-tuning**
   - learn new tasks one after another
   - no explicit forgetting mitigation

2. **replay-based continual learning**
   - mix current-task data with stored examples from earlier tasks

3. **replay + regularization**
   - combine replay with a penalty that discourages the model from drifting too far from older parameters

This gives a concise but meaningful view of the main design decisions in continual learning.

## Why this backbone is useful

Continual learning is often discussed at a high level, but the major algorithmic choices are usually variations of the same core pattern:

- how to expose the model to past data again
- how strongly to constrain new updates
- how to balance adaptation with retention

This scaffold makes those choices explicit in the smallest runnable form.

## Scope

This repository is a **foundational prototype** for continual-learning mechanics.

It does not attempt to reproduce full LLM continual-pretraining or instruction-tuning stacks.

Instead, it isolates the core algorithmic structure in a compact setting so the tradeoffs are easy to inspect.

## Requirements

- Python 3.9+
- NumPy

## Install

```bash
pip install numpy
```
