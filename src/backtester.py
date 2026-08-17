from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model import select_top_n

BASE_DIR = Path(__file__).resolve().parents[1]

RANKING_PATH = BASE_DIR / "data" / "processed" / "ranking.parquet"
IBOV_PATH = BASE_DIR / "data" / "processed" / "ibov.parquet"
OUTPUT_DIR = BASE_DIR / "reports"

TRANSACTION_COST_BPS = 10  # custo de fricção por perna, ida ou volta


def compute_portfolio_returns(ranking):
    top = select_top_n(ranking)

    top = top.sort_values(["ticker", "date"])

    # retorno do mês seguinte à seleção (rebalance com dado até o
    # fechamento do mês; resultado é o retorno do mês posterior).
    # usa o close mensal já calculado no painel de ranking, não
    # recomputa a partir dos preços diários
    prices_by_ticker_date = (
        ranking[["ticker", "date", "close"]]
        .drop_duplicates(subset=["ticker", "date"])
        .sort_values(["ticker", "date"])
    )
    prices_by_ticker_date["proximo_close"] = (
        prices_by_ticker_date.groupby("ticker")["close"].shift(-1)
    )
    prices_by_ticker_date["retorno_proximo_mes"] = (
        prices_by_ticker_date["proximo_close"]
        / prices_by_ticker_date["close"]
        - 1
    )

    top = top.merge(
        prices_by_ticker_date[["ticker", "date", "retorno_proximo_mes"]],
        on=["ticker", "date"],
        how="left",
    )

    top = top.dropna(subset=["retorno_proximo_mes"])

    top["retorno_ponderado"] = top["peso"] * top["retorno_proximo_mes"]

    portfolio_returns = (
        top
        .groupby("date", as_index=False)["retorno_ponderado"]
        .sum()
        .rename(columns={"retorno_ponderado": "portfolio_return"})
    )

    turnover = compute_turnover(top)

    portfolio_returns = portfolio_returns.merge(
        turnover, on="date", how="left"
    )
    portfolio_returns["turnover"] = portfolio_returns["turnover"].fillna(0)

    portfolio_returns["custo"] = (
        portfolio_returns["turnover"] * TRANSACTION_COST_BPS / 10_000
    )

    portfolio_returns["portfolio_return_liquido"] = (
        portfolio_returns["portfolio_return"] - portfolio_returns["custo"]
    )

    return portfolio_returns


def compute_turnover(top):
    """Fração da carteira que muda de composição a cada rebalance
    (0 = carteira idêntica ao mês anterior, 1 = trocou tudo)."""

    dates = sorted(top["date"].unique())
    records = []

    previous_holdings = set()

    for date in dates:
        current_holdings = set(
            top.loc[top["date"] == date, "ticker"]
        )

        if previous_holdings:
            changed = len(
                current_holdings.symmetric_difference(previous_holdings)
            )
            turnover = changed / (2 * len(current_holdings))
        else:
            turnover = 1.0

        records.append({"date": date, "turnover": turnover})
        previous_holdings = current_holdings

    return pd.DataFrame(records)


def compute_ibov_monthly_returns(ibov):
    ibov = ibov.sort_values("date").copy()
    ibov["month"] = ibov["date"].dt.to_period("M")

    monthly = (
        ibov
        .groupby("month", as_index=False)
        .last()
    )

    monthly["date"] = monthly["month"].dt.to_timestamp("M")
    monthly["ibov_return"] = monthly["close"].pct_change()

    return monthly[["date", "ibov_return"]].dropna()


def compute_metrics(returns):
    r = returns.dropna()
    n = len(r)

    if n == 0:
        return pd.Series({
            "observacoes": 0, "retorno_acumulado": np.nan,
            "cagr": np.nan, "volatilidade": np.nan,
            "sharpe": np.nan, "max_drawdown": np.nan,
        })

    retorno_acumulado = (1 + r).prod() - 1
    anos = n / 12

    cagr = (
        (1 + retorno_acumulado) ** (1 / anos) - 1
        if anos > 0 and (1 + retorno_acumulado) > 0
        else np.nan
    )

    vol = r.std(ddof=1) * np.sqrt(12) if n > 1 else np.nan

    sharpe = (
        r.mean() / r.std(ddof=1) * np.sqrt(12)
        if n > 1 and r.std(ddof=1) > 0
        else np.nan
    )

    patrimonio = (1 + r).cumprod()
    drawdown = patrimonio / patrimonio.cummax() - 1

    return pd.Series({
        "observacoes": n,
        "retorno_acumulado": retorno_acumulado,
        "cagr": cagr,
        "volatilidade": vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
    })


def build_comparison(portfolio_returns, ibov_returns):
    comparison = portfolio_returns.merge(
        ibov_returns, on="date", how="inner"
    )

    comparison["patrimonio_estrategia"] = (
        1 + comparison["portfolio_return_liquido"]
    ).cumprod()

    comparison["patrimonio_ibov"] = (
        1 + comparison["ibov_return"]
    ).cumprod()

    comparison["drawdown_estrategia"] = (
        comparison["patrimonio_estrategia"]
        / comparison["patrimonio_estrategia"].cummax()
        - 1
    )

    comparison["drawdown_ibov"] = (
        comparison["patrimonio_ibov"]
        / comparison["patrimonio_ibov"].cummax()
        - 1
    )

    comparison["excess_return"] = (
        comparison["portfolio_return_liquido"] - comparison["ibov_return"]
    )

    return comparison


def save_charts(comparison):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dates = comparison["date"]

    plt.figure(figsize=(12, 6))
    plt.plot(dates, comparison["patrimonio_estrategia"], label="Estratégia")
    plt.plot(dates, comparison["patrimonio_ibov"], label="IBOV")
    plt.title("Evolução do Patrimônio — Estratégia vs IBOV")
    plt.xlabel("Data")
    plt.ylabel("Patrimônio (base = 1)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curva_patrimonio.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(dates, comparison["drawdown_estrategia"], label="Estratégia")
    plt.plot(dates, comparison["drawdown_ibov"], label="IBOV")
    plt.axhline(0, linewidth=1, color="black")
    plt.title("Drawdown — Estratégia vs IBOV")
    plt.xlabel("Data")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drawdown_comparativo.png", dpi=150)
    plt.close()

    print(f"\nGráficos salvos em:\n{OUTPUT_DIR}")


def save_outputs(comparison, metrics_estrategia, metrics_ibov):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison[
        ["date", "portfolio_return_liquido", "patrimonio_estrategia"]
    ].to_csv(
        OUTPUT_DIR / "portfolio_returns_oficial.csv", index=False
    )

    comparison[
        ["date", "patrimonio_estrategia", "patrimonio_ibov"]
    ].to_csv(
        OUTPUT_DIR / "curva_patrimonio_oficial.csv", index=False
    )

    comparison[["date", "ibov_return"]].to_csv(
        OUTPUT_DIR / "benchmark_ibovespa.csv", index=False
    )

    comparison[
        ["date", "portfolio_return", "portfolio_return_liquido", "custo"]
    ].to_csv(
        OUTPUT_DIR / "impacto_custos.csv", index=False
    )

    metrics = pd.DataFrame({
        "estrategia": metrics_estrategia,
        "ibovespa": metrics_ibov,
    })
    metrics.to_csv(OUTPUT_DIR / "metricas_finais.csv")

    print(f"\nOutputs salvos em:\n{OUTPUT_DIR}")


def main():
    print("Rodando backtest")

    ranking = pd.read_parquet(RANKING_PATH)
    ibov = pd.read_parquet(IBOV_PATH)

    portfolio_returns = compute_portfolio_returns(ranking)
    ibov_returns = compute_ibov_monthly_returns(ibov)

    comparison = build_comparison(portfolio_returns, ibov_returns)

    metrics_estrategia = compute_metrics(
        comparison["portfolio_return_liquido"]
    )
    metrics_ibov = compute_metrics(comparison["ibov_return"])

    print(f"\nPeríodo: {comparison['date'].min().date()} -> "
          f"{comparison['date'].max().date()}")
    print(f"Observações: {len(comparison)}")

    print("\n=== ESTRATÉGIA ===")
    print(metrics_estrategia.to_string())

    print("\n=== IBOVESPA ===")
    print(metrics_ibov.to_string())

    print(f"\nTurnover médio: {comparison['turnover'].mean():.1%}")
    print(f"Custo médio mensal: {comparison['custo'].mean():.4%}")

    save_outputs(comparison, metrics_estrategia, metrics_ibov)
    save_charts(comparison)


if __name__ == "__main__":
    main()
