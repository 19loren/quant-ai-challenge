# Quant AI Challenge 2026 - Itaú Asset Management

This repository contains the development and results of the quantitative trading strategy designed for the **Itaú Asset Management Quant AI Challenge**. The project implements an end-to-end (E2E) data pipeline, predictive modeling, and a robust backtesting engine tailored for the Brazilian financial market.

---

## 💻 Software Architecture

The project follows strict software design patterns to ensure modularity, maintainability, and testability.

### Repository Structure
```text
├── config/
│   └── ibrx_portfolios.csv  # Carteiras teóricas do IBrX100 por período (fonte: B3)
├── data/
│   ├── raw/                 # Dados brutos — B3, CVM, IBrX100 (não versionado)
│   └── processed/           # Parquet gerados pelo pipeline
├── reports/                 # CSVs e gráficos do backtest (gerado por src/backtester.py)
├── src/
│   ├── download_cvm.py    # Baixa DFP da CVM (2010+) do portal de dados abertos
│   ├── universe.py        # Universo histórico do IBrX100 (carteiras teóricas)
│   ├── data.py             # Parsing do COTAHIST (B3) — preços e Ibovespa
│   ├── fundamentals.py     # Lucro líquido / patrimônio líquido a partir do DFP
│   ├── features.py        # Indicadores de qualidade e alavancagem (ROE, EBITDA etc.)
│   ├── ticker_mapping.py   # Pareamento ticker (B3) <-> CD_CVM (CVM)
│   ├── model.py             # Sinal multifatorial (momentum, qualidade, risco) + overlay de regime
│   ├── backtester.py       # Rebalanceamento mensal, custos, métricas vs. Ibovespa
│   └── validation.py       # Validação dos datasets processados
├── main.py                # Roda o pipeline completo, na ordem
├── requirements.txt
└── TODO.md                # Decisões e limitações conhecidas de cada etapa
```

### Como rodar

```bash
pip install -r requirements.txt
python main.py
```

Espera dados brutos já baixados em `data/raw/` (B3, CVM, IBrX100 — ver `src/download_cvm.py` para os da CVM). Roda as 8 etapas do pipeline em sequência e para no primeiro erro.

---
## 📑 Academic References 

- Jegadeesh, N. & Titman, S. (1993). *Returns to buying winners and selling losers.* Journal of Finance.
- Fama, E. & French, K. (1992). *The cross-section of expected stock returns.* Journal of Finance.
- Asness, C., Moskowitz, T. & Pedersen, L. (2013). *Value and Momentum Everywhere.* Journal of Finance.
- Novy-Marx, R. (2012). *Is momentum really momentum?* Journal of Financial Economics.

-----
