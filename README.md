#  HydroSense-Kenya: Precision Irrigation & Optimization System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/Package%20Manager-uv-purple)](https://github.com/astral-sh/uv)
[![Testing](https://img.shields.io/badge/Testing-pytest-green)](#testing)

**HydroSense-Kenya** is an algorithmic, data-driven agricultural water management system built for our Capstone Project. It transitions farming away from intuitive watering toward precision agriculture by minimizing water consumption and pumping costs while strictly adhering to the biological moisture thresholds of specific crops.

##  Project Overview
This repository contains the end-to-end mathematical pipeline for the project, spanning from raw IoT sensor data imputation to complex continuous Differential Equation (ODE) solvers, culminating in a Model Predictive Control (MPC) optimization algorithm.

### Core Mathematical Implementations:
* **Vectorized Computations (Level 2):** C-level speed arrays (via NumPy) to calculate empirical Evapotranspiration (ET) formulas and drainage parameters.
* **Numerical Methods (Level 3):** Custom implementation of **1st-Order Euler** and **4th-Order Runge-Kutta (RK4)** numerical integration methods to model continuous soil moisture flux ($\frac{dS}{dt}$).
* **Algorithmic Optimization (Level 5):** A greedy, forward-looking Model Predictive Control algorithm that simulates future environmental states to calculate the exact optimal millimeter of irrigation required to prevent crop stress.

---

##  Repository Architecture

To maintain strict separation of concerns, the mathematical engines are decoupled from the academic analysis layer. 

```text
hydrosense-kenya/
│
├── data/
│   ├── raw/                 # Raw IoT, weather, and crop parameters
│   └── processed/           # Checkpoints & clean pipeline outputs
│
├── notebooks/               # Academic Presentation & Data Visualization
│   ├── Level_1_Problem_Framing.ipynb
│   ├── Level_2_Vectorization_and_Error.ipynb
│   ├── Level_3_Numerical_Methods.ipynb
│   ├── Level_4_Data_Analysis_and_Visualization.ipynb
│   ├── Level_5_Simulation_and_Optimization.ipynb
│   └── Level_6_Final_Integration.ipynb
│
├── src/                     # Core Mathematical Engines
│   ├── data_cleaning.py     # ETL and Imputation
│   ├── numerical_methods.py # ODE Integrators (Euler / RK4)
│   └── simulation.py        # Optimization & Predictive Control
│
├── tests/                   # Pytest Quality Control Suite
│   ├── test_integration.py
│   └── test_simulation.py
│
├── main.py                  # Single-command end-to-end execution pipeline
├── pyproject.toml           # Project metadata and dependencies
└── uv.lock                  # Deterministic lockfile