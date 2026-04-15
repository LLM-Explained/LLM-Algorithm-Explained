# Muon Optimizer Backbone

This repository provides a **core backbone** for the main algorithmic idea behind **Muon**:

for hidden-layer matrix weights, optimize the **geometry of the whole matrix update** rather than only applying coordinate-wise scaling.

## What this code covers

This scaffold includes the major algorithmic pieces:

- **momentum accumulation**
- **matrix orthogonalization**
- **Newton–Schulz approximation** of the orthogonalized update
- **Muon update on hidden-layer 2D weights**
- **standard optimizer updates** for biases / output layer

## Core equations

Momentum update:

```latex
M_t = \mu M_{t-1} + \nabla L_t(W_{t-1})
```
