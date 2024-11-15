# Value At Risk Calculator For Your Portfolio

This project implements a Monte Carlo simulation-based Value at Risk (VaR) calculator for portfolio risk assessment. The `MonteCarlo` class simulates portfolio returns over time and calculates the potential loss for a given confidence level.

## Overview

The Monte Carlo Value at Risk (VaR) technique estimates the potential loss in a portfolio over a specified time horizon and confidence interval. This implementation:

- Simulates daily portfolio returns based on mean returns and a covariance matrix.
- Calculates portfolio values over multiple simulations.
- Computes the Value at Risk (VaR) at a specified confidence interval.

## Features

- **Monte Carlo Simulation**: Generates multiple portfolio return scenarios over a given number of days.
- **Portfolio Value Simulation**: Tracks portfolio value over time for each simulation.
- **VaR Calculation**: Calculates Value at Risk based on the Monte Carlo simulation results.

## Requirements

- Python 3.x
- `numpy`
- `pandas`
- `matplotlib
