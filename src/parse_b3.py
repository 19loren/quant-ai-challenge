'''
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

B3_DIR = BASE_DIR / "data" / "raw" / "b3_quotes"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "b3_prices"


def parse_b3_file(filepath):
    records = []

    with open(filepath, "r", encoding="latin-1") as f:

        # header
        f.readline()

        for line in f:
            line = line.rstrip("\r\n")

            # apenas registros de negociaçao
            if line[0:2] != "01":
                continue

            records.append(
                {
                    "date": pd.to_datetime(
                        line[2:10],
                        format="%Y%m%d"
                    ),
                    "ticker": line[12:24].strip(),
                    "market": line[24:27].strip(),
                    "name": line[27:39].strip(),
                    "security": line[39:49].strip(),
                    "open": int(line[56:69]) / 100,
                    "high": int(line[69:82]) / 100,
                    "low": int(line[82:95]) / 100,
                    "average": int(line[95:108]) / 100,
                    "close": int(line[108:121]) / 100,
                    "trades": int(line[147:152]),
                    "volume": int(line[152:170]),
                }
            )

    return pd.DataFrame(records)


if __name__ == "__main__":

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    years = range(2019, 2026)

    for year in years:

        filepath = B3_DIR / f"COTAHIST_A{year}.TXT"
        output_path = OUTPUT_DIR / f"prices_{year}.parquet"

        print()
        print("=" * 60)
        print(f"PROCESSANDO {year}")
        print("=" * 60)

        if not filepath.exists():
            print(f"Arquivo não encontrado: {filepath}")
            continue

        print(f"Lendo: {filepath.name}")

        df = parse_b3_file(filepath)

        print(f"Linhas: {len(df):,}")
        print(
            f"Período: "
            f"{df['date'].min().date()} → "
            f"{df['date'].max().date()}"
        )

        print(f"Salvando: {output_path}")

        df.to_parquet(
            output_path,
            index=False
        )

        print("OK")

        del df

    print()
    print("=" * 60)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 60)
'''
from pathlib import Path
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "data" / "processed" / "b3_prices"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "prices.parquet"


files = sorted(INPUT_DIR.glob("prices_*.parquet"))

if not files:
    raise FileNotFoundError(
        "Nenhum arquivo prices_*.parquet encontrado."
    )


print("=" * 60)
print("CONSOLIDANDO PREÇOS")
print("=" * 60)

print(f"\nArquivos encontrados: {len(files)}")

for file in files:
    metadata = pq.ParquetFile(file).metadata
    print(
        f"{file.name}: "
        f"{metadata.num_rows:,} linhas"
    )


# Usa o schema do primeiro arquivo
first_file = pq.ParquetFile(files[0])
schema = first_file.schema_arrow

writer = pq.ParquetWriter(
    OUTPUT_PATH,
    schema
)

total_rows = 0

try:

    for file in files:

        print(f"\nProcessando: {file.name}")

        parquet_file = pq.ParquetFile(file)

        for row_group in range(
            parquet_file.num_row_groups
        ):

            table = parquet_file.read_row_group(
                row_group
            )

            writer.write_table(table)

            total_rows += table.num_rows

        print("OK")

finally:
    writer.close()


print()
print("=" * 60)
print("CONSOLIDAÇÃO CONCLUÍDA")
print("=" * 60)

print(f"\nArquivo: {OUTPUT_PATH}")
print(f"Linhas: {total_rows:,}")