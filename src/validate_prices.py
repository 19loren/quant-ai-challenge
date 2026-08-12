from pathlib import Path
import pandas as pd


PRICES_PATH = Path("data/processed/prices.parquet")


prices = pd.read_parquet(PRICES_PATH)


print("FORMATO DA BASE:")
print(prices.shape)


print("\nPERÍODO:")
print("Início:", prices["date"].min())
print("Fim:", prices["date"].max())


print("\nATIVOS:")
print("Quantidade:", prices["ticker"].nunique())


print("\nVALORES AUSENTES:")
print(prices.isna().sum())


print("\nDUPLICATAS:")
print(
    prices.duplicated(
        subset=["date", "ticker"]
    ).sum()
)


print("\nEXEMPLO DE UM ATIVO:")
example_ticker = prices["ticker"].iloc[0]

print(
    prices[
        prices["ticker"] == example_ticker
    ].head(10)
)