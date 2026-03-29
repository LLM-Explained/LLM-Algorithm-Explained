# Markov State Abstraction Toy Demo

This is a minimal runnable demo illustrating the core intuition behind:

**Breaking the Capability Ceiling of LLM Post-Training by Reintroducing Markov States**

## What this demo shows

It compares two tiny Q-learning agents on a toy environment:

1. **history-as-state**
   - the agent treats the whole growing trajectory history as the state

2. **Markov-state**
   - the agent uses a compact decision state `(position, key_collected)`

The main idea is to show why compact state abstraction can improve learning efficiency and reuse of experience.

## What this demo is

A tiny educational analogy for the paper's central idea.

## What this demo is NOT

This is not an implementation of the paper.

It does **not** include:
- LLM post-training
- policy gradients
- reasoning traces
- reward modeling
- language-based state abstraction
- the actual benchmark tasks from the paper

It only demonstrates the state-abstraction intuition in the smallest runnable form.

## Requirements

- Python 3.9+

No extra dependencies are required.

## Run

```bash
python demo.py
```
