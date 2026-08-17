from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

PRICES_PATH = BASE_DIR / "data" / "processed" / "prices.parquet"
IBOV_PATH = BASE_DIR / "data" / "processed" / "ibov.parquet"
UNIVERSE_PATH = BASE_DIR / "data" / "processed" / "universe.parquet"
QUALITY_PATH = BASE_DIR / "data" / "processed" / "quality_features.parquet"
TICKER_CVM_MAP_PATH = BASE_DIR / "data" / "processed" / "ticker_cvm_map.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "ranking.parquet"

# a especificação pede Ibovespa vs. média móvel de 200 pregões, mas
# o COTAHIST registra o IBOV em frequência irregular — ~1x/mês até
# 2024, quase diário a partir de dez/2025 (ver TODO.md). Uma janela
# por contagem de linhas ficaria ora ~10 meses, ora ~2 semanas —
# por isso a janela é por tempo corrido (~200 pregões ≈ 290 dias
# corridos, com fins de semana/feriados)
REGIME_WINDOW = "290D"
REGIME_MIN_OBSERVATIONS = 6

MOMENTUM_LAG_SHORT = 21
MOMENTUM_LAG_LONG = 252
VOLATILITY_WINDOW = 60

# pesos por regime: em mercado favorável, mais peso pra momentum;
# em estresse, migra peso pra qualidade e menor risco
REGIME_WEIGHTS = {
    "favoravel": {
        "momentum": 0.50,
        "quality": 0.30,
        "leverage": 0.10,
        "risk": 0.10,
    },
    "estresse": {
        "momentum": 0.20,
        "quality": 0.40,
        "leverage": 0.15,
        "risk": 0.25,
    },
}

TOP_N = 10

EMBARGO_DAYS = 90


def winsorize(s, lower=0.05, upper=0.95):
    if s.notna().sum() < 5:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def zscore(s):
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def compute_market_regime(ibov):
    ibov = ibov.sort_values("date").copy()

    rolling = (
        ibov
        .set_index("date")["close"]
        .rolling(REGIME_WINDOW, min_periods=REGIME_MIN_OBSERVATIONS)
        .mean()
    )

    ibov["sma"] = rolling.values

    ibov["regime"] = np.where(
        ibov["close"] >= ibov["sma"],
        "favoravel",
        "estresse",
    )

    # antes de ter janela suficiente, assume regime favorável
    # (postura neutra/padrão, não há sinal de estresse ainda)
    ibov["regime"] = ibov["regime"].where(
        ibov["sma"].notna(), "favoravel"
    )

    return ibov[["date", "close", "sma", "regime"]]


def compute_price_signals(prices, universe_tickers):
    px = prices[
        prices["ticker"].isin(universe_tickers)
    ][["date", "ticker", "close"]].copy()

    px = px.sort_values(["ticker", "date"])

    px["ret"] = px.groupby("ticker")["close"].pct_change()

    px["close_lag_short"] = (
        px.groupby("ticker")["close"].shift(MOMENTUM_LAG_SHORT)
    )
    px["close_lag_long"] = (
        px.groupby("ticker")["close"].shift(MOMENTUM_LAG_LONG)
    )

    px["momentum_12_1"] = (
        px["close_lag_short"] / px["close_lag_long"] - 1
    )

    px["vol_60"] = (
        px.groupby("ticker")["ret"]
        .rolling(VOLATILITY_WINDOW)
        .std()
        .reset_index(level=0, drop=True)
        * (252 ** 0.5)
    )

    px["month"] = px["date"].dt.to_period("M")

    monthly = (
        px
        .sort_values("date")
        .groupby(["ticker", "month"], as_index=False)
        .last()
    )

    monthly["date"] = monthly["month"].dt.to_timestamp("M")

    return monthly[
        ["ticker", "date", "close", "momentum_12_1", "vol_60"]
    ]


def attach_point_in_time_quality(monthly_px, quality):
    """Para cada (ticker, data de rebalance), usa o indicador
    fundamentalista mais recente cujo DT_FIM_EXERC já tinha sido
    divulgado — embargo de 90 dias, evita look-ahead bias."""

    quality = quality.dropna(subset=["CD_CVM"]).copy()
    quality["CD_CVM"] = quality["CD_CVM"].astype("Int64")
    quality = quality.sort_values("DT_FIM_EXERC")

    dates = monthly_px[["date"]].drop_duplicates().sort_values("date")

    records = []

    for cd_cvm, group in quality.groupby("CD_CVM"):
        group = group.sort_values("DT_FIM_EXERC")

        for date in dates["date"]:
            available = group[
                group["DT_FIM_EXERC"]
                <= date - pd.Timedelta(days=EMBARGO_DAYS)
            ]

            if available.empty:
                continue

            row = available.iloc[-1]

            records.append(
                {
                    "CD_CVM": cd_cvm,
                    "date": date,
                    "quality_score": row["quality_score"],
                    "leverage_score": row["leverage_score"],
                }
            )

    return pd.DataFrame(records)


def build_ranking():
    prices = pd.read_parquet(PRICES_PATH)
    ibov = pd.read_parquet(IBOV_PATH)
    universe = pd.read_parquet(UNIVERSE_PATH)
    quality = pd.read_parquet(QUALITY_PATH)

    regime = compute_market_regime(ibov)

    universe_tickers = universe["ticker"].unique()
    monthly_px = compute_price_signals(prices, universe_tickers)

    quality_pit = attach_point_in_time_quality(monthly_px, quality)

    if not TICKER_CVM_MAP_PATH.exists():
        raise FileNotFoundError(
            "ticker_cvm_map.parquet não encontrado — rode "
            "`python -m src.ticker_mapping` primeiro."
        )

    ticker_cvm = pd.read_parquet(TICKER_CVM_MAP_PATH)[
        ["ticker", "CD_CVM"]
    ].drop_duplicates()

    panel = monthly_px.merge(ticker_cvm, on="ticker", how="left")
    panel = panel.merge(
        quality_pit, on=["CD_CVM", "date"], how="left"
    )

    panel["month"] = panel["date"].dt.to_period("M")
    regime["month"] = pd.to_datetime(regime["date"]).dt.to_period("M")

    panel = panel.merge(
        regime[["month", "regime"]], on="month", how="left"
    )
    panel["regime"] = panel["regime"].fillna("favoravel")

    for raw, winsorized in [
        ("momentum_12_1", "momentum_w"),
        ("quality_score", "quality_w"),
        ("leverage_score", "leverage_w"),
        ("vol_60", "risk_w"),
    ]:
        panel[winsorized] = (
            panel.groupby("date")[raw].transform(winsorize)
        )

    for winsorized, z in [
        ("momentum_w", "z_momentum"),
        ("quality_w", "z_quality"),
        ("leverage_w", "z_leverage"),
        ("risk_w", "z_risk"),
    ]:
        panel[z] = panel.groupby("date")[winsorized].transform(zscore)

    # menor volatilidade = melhor
    panel["z_risk"] = -panel["z_risk"]

    weights = panel["regime"].map(REGIME_WEIGHTS)

    panel["factor_score"] = (
        panel["z_momentum"] * weights.apply(lambda w: w["momentum"])
        + panel["z_quality"] * weights.apply(lambda w: w["quality"])
        + panel["z_leverage"] * weights.apply(lambda w: w["leverage"])
        + panel["z_risk"] * weights.apply(lambda w: w["risk"])
    )

    panel["rank"] = (
        panel
        .dropna(subset=["factor_score"])
        .groupby("date")["factor_score"]
        .rank(ascending=False, method="first")
    )

    return panel


def select_top_n(ranking):
    top = (
        ranking
        .dropna(subset=["rank"])
        .query("rank <= @TOP_N")
        .sort_values(["date", "rank"])
        .copy()
    )

    top["peso"] = 1 / TOP_N

    return top


def save_ranking(ranking):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nArquivo salvo em:\n{OUTPUT_PATH}")
    print(f"Linhas salvas: {len(ranking):,}")


def main():
    print("Construindo ranking multifatorial")

    ranking = build_ranking()

    print(f"\nPainel: {ranking.shape}")
    print(f"Datas de rebalanceamento: {ranking['date'].nunique()}")

    print("\nDistribuição de regime por mês:")
    print(
        ranking
        .drop_duplicates("date")["regime"]
        .value_counts()
    )

    top = select_top_n(ranking)
    print(f"\nCarteiras top-{TOP_N}: {top['date'].nunique()} datas")

    save_ranking(ranking)


if __name__ == "__main__":
    main()
