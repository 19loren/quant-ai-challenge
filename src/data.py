from pathlib import Path
import pandas as pd


COTAHIST_PATH = Path("data/raw/b3_quotes/COTAHIST_A2025.TXT")


# lista para armazenar as cotações
records = []


with open(COTAHIST_PATH, "r", encoding="latin-1") as file:

    # ignora o header
    file.readline()

    for line in file:

        # ignora linhas vazias
        if not line.strip():
            continue

        # tipo de registro
        record_type = line[0:2]

        # processa apenas registros de negociação
        if record_type != "01":
            continue

        # extrai campos básicos
        date = line[2:10]
        ticker = line[12:24].strip()
        market_type = line[24:27].strip()
        name = line[27:39].strip()
        security_type = line[39:49].strip()

        # mantem apenas o mercado à vista
        if market_type != "010":
             continue

        # mantem apenas açoes e units
        security_base = security_type.split()[0]
        if security_base not in {"ON", "PN", "PNA", "PNB", "UNT"}:
            continue
        
        # preços
        open_raw = line[56:69].strip()
        high_raw = line[69:82].strip()
        low_raw = line[82:95].strip()
        avg_raw = line[95:108].strip()
        close_raw = line[108:121].strip()

        # ignora registros sem preço
        if not close_raw:
            continue

        # converte preços
        open_price = float(open_raw) / 100 if open_raw else None
        high_price = float(high_raw) / 100 if high_raw else None
        low_price = float(low_raw) / 100 if low_raw else None
        avg_price = float(avg_raw) / 100 if avg_raw else None
        close_price = float(close_raw) / 100

        # adiciona registro
        records.append({
            "date": date,
            "ticker": ticker,
            "market_type": market_type,
            "name": name,
            "security_type": security_type,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "avg": avg_price,
            "close": close_price,
        })


prices = pd.DataFrame(records)

# converte data
prices["date"] = pd.to_datetime(
    prices["date"],
    format="%Y%m%d"
)

# ordena
prices = prices.sort_values(
    ["ticker", "date"]
).reset_index(drop=True)

# informações do dataset
print("Quantidade de registros:", len(prices))
print("Quantidade de ativos:", prices["ticker"].nunique())

print("\nPRIMEIROS REGISTROS:")
print(prices.head())

print("\nTIPOS DAS COLUNAS:")
print(prices.dtypes)
print("\nTIPOS DE ATIVOS:")
print(
    prices["security_type"]
    .value_counts()
    .head(30)
)
print("\nTIPOS DE MERCADO:")
print(
    prices["market_type"]
    .value_counts()
)

# salva a base processada
OUTPUT_PATH = Path("data/processed/prices.parquet")

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

prices.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(f"\nArquivo salvo em: {OUTPUT_PATH}")
print(f"Linhas salvas: {len(prices)}")