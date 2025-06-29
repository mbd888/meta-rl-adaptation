# Context-Conditioned Meta-Reinforcement Learning on CartPole

This project implements a context-based meta-reinforcement learning (meta-RL) system for zero-shot adaptation across variants of the CartPole environment. It uses a modular architecture comprising a trajectory-based context encoder and a context-conditioned policy network, enabling fast policy adaptation without test-time gradient updates.

## Project Summary

- **Environment:** Modified `CartPole` with dynamics variation (light, medium, heavy poles).
- **Architecture:**
  - **Context Encoder**: Converts a short trajectory (s, a, r sequences) into a latent context vector `z`.
  - **Policy Network**: Receives `(s, z)` and outputs action `a` to balance the pole.
- **Training:** Uses the REINFORCE algorithm across multiple task variants.
- **Adaptation:** Achieves zero-shot generalization by inferring task characteristics from limited past interactions.

## Paper

You can read the full technical writeup [here (PDF)](./report/context_meta_rl_cartpole.pdf).

> The report includes system design, empirical evaluation, analysis of generalization behavior, and proposed improvements such as probabilistic context encoding, exploration strategies, and scaling to high-dimensional tasks.

## Visualizations

- Architecture diagram (context encoder + policy)
- CartPole variants (light, medium, heavy)
- Context latent space (PCA/t-SNE)
- Performance metrics (returns on seen/unseen tasks)

## Requirements

- Python 3.8+
- PyTorch
- OpenAI Gym (`CartPole-v1`)
- NumPy, Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
