# Conditional Group Clustering via Genetic Algorithm

This repository provides a framework for clustering individuals into groups while satisfying a complex set of predefined conditions and constraints.

Manually creating optimized groups is a difficult combinatorial problem. Even with a small number of people and a few constraints (e.g., "Person A cannot be with Person B," "Each group must have one expert"), the number of possible combinations becomes computationally unfeasible to check exhaustively.

This project uses a **Genetic Algorithm (GA)** to efficiently search the solution space and find high-quality groupings that best satisfy the specified rules. It is built on top of the [DEAP](https://github.com/DEAP/deap) library.

## ✨ Features

* **Constraint-based Clustering:** Define complex rules, "hard" (must-have) and "soft" (nice-to-have) conditions, and penalties for group formation.
* **Genetic Algorithm Engine:** Utilizes GA (via DEAP) to evolve solutions and find optimal or near-optimal groupings.
* **Configurable:** Easily define your dataset, constraints, and GA parameters using YAML configuration files.
* **Results Analysis:** Includes tools to analyze and report on the quality of the generated groups.

## 🚀 Installation

### Requirements
You can install all requirements using the provided file:

```bash
pip install -r requirements.txt

