from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

PRICES_PATH = (BASE_DIR / "data" / "processed" / "prices.parquet")

UNIVERSE_PATH = (BASE_DIR / "data" / "processed" / "universe.parquet")


def validate_prices(prices):

    print("\nValidação dos preços")

    print(
        f"Dimensões: {prices.shape}"
    )

    print(
        f"Período: "
        f"{prices['date'].min()} → "
        f"{prices['date'].max()}"
    )

    print(
        f"Ativos únicos: "
        f"{prices['ticker'].nunique()}"
    )

    missing_values = (
        prices.isna().sum()
    )

    print("\nValores ausentes:")

    print(
        missing_values[
            missing_values > 0
        ]
    )

    duplicate_count = (
        prices
        .duplicated(
            subset=["date", "ticker"]
        )
        .sum()
    )

    print(
        f"\nDuplicidades "
        f"(date, ticker): {duplicate_count}"
    )

    if duplicate_count > 0:
        raise ValueError(
            "Existem registros duplicados "
            "para a combinação date/ticker."
        )

    if prices["date"].isna().any():
        raise ValueError(
            "Existem datas ausentes."
        )

    if prices["ticker"].isna().any():
        raise ValueError(
            "Existem tickers ausentes."
        )

    if (prices["close"] <= 0).any():
        raise ValueError(
            "Existem preços de fechamento "
            "menores ou iguais a zero."
        )

    print("Validação dos preços concluída.")


def validate_universe(universe):

    print("\nValidação do universo")

    print(
        f"Dimensões: {universe.shape}"
    )

    print(
        f"Períodos: "
        f"{universe['period_start'].nunique()}"
    )

    print(
        f"Tickers históricos: "
        f"{universe['ticker'].nunique()}"
    )

    duplicate_count = (
        universe
        .duplicated(
            subset=["period_start", "ticker"]
        )
        .sum()
    )

    print(
        f"\nDuplicidades "
        f"(period_start, ticker): "
        f"{duplicate_count}"
    )

    if duplicate_count > 0:
        raise ValueError(
            "Existem tickers duplicados "
            "dentro de um período."
        )

    if universe["ticker"].isna().any():
        raise ValueError(
            "Existem tickers ausentes."
        )

    if universe["period_start"].isna().any():
        raise ValueError(
            "Existem períodos iniciais ausentes."
        )

    if universe["period_end"].isna().any():
        raise ValueError(
            "Existem períodos finais ausentes."
        )

    if universe["base_date"].isna().any():
        raise ValueError(
            "Existem base dates ausentes."
        )

    counts = (
        universe
        .groupby("period_start")["ticker"]
        .nunique()
    )

    print("\nAtivos por período:")

    print(counts)

    if (counts <= 0).any():
        raise ValueError(
            "Existe um período sem ativos."
        )

    print(
        "Validação do universo concluída."
    )


# PROVISORIO
def main():

    prices = pd.read_parquet(
        PRICES_PATH
    )

    universe = pd.read_parquet(
        UNIVERSE_PATH
    )

    validate_prices(
        prices
    )

    validate_universe(
        universe
    )

    print(
        "\nTodas as validações concluídas."
    )


if __name__ == "__main__":
    main()