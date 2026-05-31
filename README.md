# Quant AI Challenge 2026 - Itaú Asset

This repository contains the development and results of the quantitative trading strategy designed for the **Itaú Asset Management Quant AI Challenge**. The project implements an end-to-end (E2E) data pipeline, predictive modeling, and a robust backtesting engine tailored for the Brazilian financial market.

---

## 💻 Software Architecture

The project follows strict software design patterns to ensure modularity, maintainability, and testability.

### Repository Structure
```text
├── data/                  # Datasets (raw and engineered files via git-lfs)
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # ETL pipeline for data ingestion, cleaning, and integrity
│   ├── features.py       # Mathematical feature engineering pipeline
│   ├── model.py          # Model training, cross-validation, and hyperparameter tuning
│   └── backtester.py     # Vectorized/Event-driven execution engine with friction costs
├── tests/                 # Unit tests for statistical formulas and risk metrics
├── config.yaml            # Global parameters, paths, and model hyperparameters
├── main.py                # Main entry point to run the entire pipeline
└── requirements.txt       # Frozen project dependencies and versions
```
---
## 📑 Academic References 

- Jegadeesh, N. & Titman, S. (1993). *Returns to buying winners and selling losers.* Journal of Finance.
- Fama, E. & French, K. (1992). *The cross-section of expected stock returns.* Journal of Finance.
- Asness, C., Moskowitz, T. & Pedersen, L. (2013). *Value and Momentum Everywhere.* Journal of Finance.
- Novy-Marx, R. (2012). *Is momentum really momentum?* Journal of Financial Economics.

-----
