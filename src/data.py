from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

COTAHIST_DIR = BASE_DIR / "data" / "raw" / "b3_quotes"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "prices.parquet"
IBOV_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "ibov.parquet"

# config
VALID_MARKET_TYPE = "010"

VALID_SECURITY_TYPES = {
    "ON",
    "PN",
    "PNA",
    "PNB",
    "UNT",
}

# o Ibovespa aparece no COTAHIST como um "ativo" (ticker IBOV11) com
# security_type "IBO/", que não passa no filtro de ações acima —
# por isso é capturado à parte, no mesmo passe de leitura
INDEX_TICKERS = {
    "IBOV11",
}

def parse_cotahist_file(filepath):

    records = []
    index_records = []
    zero_close_count = 0

    print(f"\nProcessando arquivo: {filepath.name}")

    with open(
        filepath,
        "r",
        encoding="latin-1"
    ) as file:

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

            if market_type != VALID_MARKET_TYPE:
                continue

            # indices (ex.: IBOV11) não têm security_type de ação,
            # então são separados antes do filtro abaixo
            if ticker in INDEX_TICKERS:
                index_close_raw = line[108:121].strip()

                if not index_close_raw:
                    continue

                index_records.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "name": name,
                        "close": float(index_close_raw) / 100,
                    }
                )

                continue

            # mantem apenas açoes e units
            security_base = security_type.split()[0]
            if security_base not in VALID_SECURITY_TYPES:
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

            # alguns pregões de negócio único vêm com OHLC zerado na
            # própria fonte (só o preço médio é preenchido) — sem
            # fechamento não há preço de referência utilizável
            if close_price <= 0:
                zero_close_count += 1
                continue

            # liquidez
            trade_raw = line[147:152].strip()
            quantity_raw = line[152:170].strip()
            volume_raw = line[170:188].strip()

            trades = (
                int(trade_raw)
                if trade_raw
                else 0
            )

            quantity = (
                int(quantity_raw)
                if quantity_raw
                else 0
            )

            financial_volume = (
                float(volume_raw) / 100
                if volume_raw
                else 0
            )

            # adiciona registro
            records.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "market_type": market_type,
                    "name": name,
                    "security_type": security_type,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "avg_price": avg_price,
                    "close": close_price,
                    "trades": trades,
                    "quantity": quantity,
                    "financial_volume": financial_volume,
                }
            )

    if zero_close_count > 0:
        print(
            f"Registros descartados por fechamento zerado: "
            f"{zero_close_count}"
        )

    return pd.DataFrame(records), pd.DataFrame(index_records)

# le todos os arquivos encontrados na b3 e retorna o preço
def load_prices():

    files = sorted(
        COTAHIST_DIR.glob("COTAHIST_A*.TXT")
    )

    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo COTAHIST encontrado em: "
            f"{COTAHIST_DIR}"
        )

    print("PIPELINE DE PREÇOS B3")
    print(f"\nArquivos encontrados: {len(files)}")

    for file in files:
        print(f" - {file.name}")

    all_prices = []
    all_index = []

    for file in files:
        df, index_df = parse_cotahist_file(file)

        print(f"Registros elegíveis: {len(df):,}")

        all_prices.append(df)
        all_index.append(index_df)

    prices = pd.concat(
        all_prices,
        ignore_index=True
    )

    index_prices = pd.concat(
        all_index,
        ignore_index=True
    )

    return prices, index_prices

# padronizacao
def standardize_prices(prices):

    prices = prices.copy()

    prices["date"] = pd.to_datetime(
    prices["date"],
    format="%Y%m%d"
)

    prices = (
        prices.sort_values(
            ["ticker", "date"]
        ).reset_index(drop=True)
    ) 

    return prices

def save_prices(prices):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    prices.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nArquivo salvo em:\n{OUTPUT_PATH}")
    print(f"Linhas salvas: {len(prices):,}")


def save_ibov(index_prices):

    IBOV_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    index_prices.to_parquet(
        IBOV_OUTPUT_PATH,
        index=False
    )

    print(f"\nArquivo salvo em:\n{IBOV_OUTPUT_PATH}")
    print(f"Linhas salvas: {len(index_prices):,}")


# MAIN POR ENQUANTO
def main():

    prices, index_prices = load_prices()
    prices = standardize_prices(prices)
    index_prices = standardize_prices(index_prices)

    print("RESULTADO")
    print(
        f"\nQuantidade de registros: "
        f"{len(prices):,}"
    )

    print(
        f"Quantidade de ativos: "
        f"{prices['ticker'].nunique():,}"
    )

    print(
        f"Período: "
        f"{prices['date'].min().date()} "
        f"-> "
        f"{prices['date'].max().date()}"
    )

    print("\nPrimeiros registros:")
    print(
        prices.head()
        .to_string(index=False)
    )

    print("\nTipos das colunas:")
    print(prices.dtypes)

    save_prices(prices)

    print(
        f"\nRegistros do Ibovespa: "
        f"{len(index_prices):,}"
    )

    save_ibov(index_prices)

if __name__ == "__main__":
    main()