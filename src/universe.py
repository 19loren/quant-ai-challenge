from pathlib import Path
import re
import pandas as pd
import pymupdf

BASE_DIR = Path(__file__).resolve().parents[1]

IBRX_DIR = BASE_DIR / "data" / "raw" / "ibrx100"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "universe.parquet"

# config das carteiras
PORTFOLIOS = [
    {
        "file": "jan_abr_2025.xlsx",
        "type": "excel",
        "start": "2025-01-06",
        "end": "2025-05-02",
        "base_date": "2025-01-03",
    },
    {
        "file": "maio_ago_2025.xlsx",
        "type": "excel",
        "start": "2025-05-05",
        "end": "2025-08-29",
        "base_date": "2025-05-02",
    },
    {
        "file": "VIRADA - SET2025.xlsx",
        "type": "excel",
        "start": "2025-09-01",
        "end": "2026-01-02",
        "base_date": "2025-08-29",
    },
    {
        "file": "jan_abr_2026.xlsx",
        "type": "excel",
        "start": "2026-01-05",
        "end": "2026-04-30",
        "base_date": "2026-01-02",
    },
    {
        "file": "maio_ago2026.pdf",
        "type": "pdf",
        "start": "2026-05-04",
        "end": "2026-09-04",
        "base_date": "2026-04-30",
    },
]

# ler ibxx excel
def read_ibrx_excel(path):

    print(f"\nLendo Excel: {path.name}")

    df = pd.read_excel(
        path,
        sheet_name="IBXX",
        header=1
    )

    # remove espaços dos nomes das colunas
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # mantem somente linhas que possuem ticker
    df = df[
        df["CÓDIGO"].notna()
    ].copy()

    # remove possiveis linhas de total/redutor
    df = df[
        df["CÓDIGO"].astype(str).str.strip().str.match(
            r"^[A-Z]{4}\d{1,2}$",
            na=False
        )
    ].copy()

    # padroniza ticker
    df["ticker"] = (
        df["CÓDIGO"]
        .astype(str)
        .str.strip()
    )

    # padroniza nome
    df["name"] = (
        df["AÇÃO"]
        .astype(str)
        .str.strip()
    )

    # padroniza tipo
    df["security_type"] = (
        df["TIPO"]
        .astype(str)
        .str.strip()
    )

    # quantidade teorica
    df["theoretical_quantity"] = pd.to_numeric(
        df["QTDE. TEÓRICA"],
        errors="coerce"
    )

    # participaçao
    df["index_weight"] = pd.to_numeric(
        df["PART. %"],
        errors="coerce"
    )

    # mantem somente as colunas necessarias
    df = df[
        [
            "ticker",
            "name",
            "security_type",
            "theoretical_quantity",
            "index_weight",
        ]
    ].copy()

    return df

# ler ibxx pdf
def read_ibrx_pdf(path):

    print(f"\nLendo PDF: {path.name}")

    pdf = pymupdf.open(path)

    # pag 7 = índice 6
    # pag 8 = índice 7
    text = ""

    for page_number in [6, 7]:
        text += pdf[page_number].get_text()
        text += "\n"

    pdf.close()

    # localiza inicio do ibxx
    start_marker = "IBXX"

    # o proximo indice começa com iee
    end_marker = "IEE"

    start = text.find(start_marker)
    end = text.find(end_marker, start + len(start_marker))

    if start == -1:
        raise ValueError(
            "Não foi possível encontrar 'IBXX' no PDF."
        )

    if end == -1:
        raise ValueError(
            "Não foi possível encontrar o início do próximo índice 'IEE'."
        )

    ibxx_text = text[
        start + len(start_marker):end
    ]

    lines = [
        line.strip()
        for line in ibxx_text.splitlines()
        if line.strip()
    ]

    # remove cabeçalhos repetidos
    header_lines = {
        "Código IF",
        "Código",
        "Ação",
        "Tipo",
        "Quantidade teórica",
        "Participação (%)",
        "Boletim Diário do Mercado",
        "Indicadores e informativos",
    }

    lines = [
        line
        for line in lines
        if line not in header_lines
    ]

    records = []

    ticker_pattern = re.compile(
        r"^[A-Z]{4}\d{1,2}$"
    )

    i = 0

    while i < len(lines):

        # procura o proximo ticker
        if not ticker_pattern.match(lines[i]):
            i += 1
            continue

        ticker = lines[i]

        if i + 4 >= len(lines):
            break

        name = lines[i + 1]
        security_type = lines[i + 2]
        quantity_raw = lines[i + 3]
        weight_raw = lines[i + 4]

        # confirma se quantidade é realmente numerica
        quantity_clean = (
            quantity_raw
            .replace(".", "")
            .replace(",", ".")
        )

        # confirma se peso é numerico
        weight_clean = (
            weight_raw
            .replace(".", "")
            .replace(",", ".")
        )

        try:
            theoretical_quantity = float(
                quantity_clean
            )

            index_weight = float(
                weight_clean
            )

        except ValueError:
            i += 1
            continue

        records.append(
            {
                "ticker": ticker,
                "name": name,
                "security_type": security_type,
                "theoretical_quantity": theoretical_quantity,
                "index_weight": index_weight,
            }
        )

        i += 5

    df = pd.DataFrame(records)

    return df

# validaçao de cada carteira
def validate_portfolio(df, period_name):
    """
    Valida se uma carteira possui exatamente
    100 ativos únicos.
    """

    print(
        f"\nValidação da carteira: {period_name}"
    )

    print(
        "Quantidade de registros:",
        len(df)
    )

    print(
        "Quantidade de tickers únicos:",
        df["ticker"].nunique()
    )

    # deve haver exatamente 100 ativos
    if len(df) != 100:
        raise ValueError(
            f"{period_name}: esperado 100 ativos, "
            f"mas foram encontrados {len(df)}."
        )

    # nao pode haver ticker duplicado
    if df["ticker"].nunique() != 100:
        duplicates = (
            df[
                df["ticker"].duplicated(keep=False)
            ]["ticker"]
            .tolist()
        )

        raise ValueError(
            f"{period_name}: existem tickers duplicados: "
            f"{duplicates}"
        )

    # quantidade teorica deve existir
    if df["theoretical_quantity"].isna().any():
        raise ValueError(
            f"{period_name}: existem quantidades "
            "teóricas ausentes."
        )

    # peso deve existir
    if df["index_weight"].isna().any():
        raise ValueError(
            f"{period_name}: existem pesos "
            "ausentes."
        )

    print("✓ 100 ativos únicos")
    print("✓ Quantidades teóricas válidas")
    print("✓ Pesos válidos")


# processamento
all_portfolios = []


for portfolio in PORTFOLIOS:

    file_path = IBRX_DIR / portfolio["file"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {file_path}"
        )

    # excel
    if portfolio["type"] == "excel":

        df = read_ibrx_excel(
            file_path
        )

    # pdf
    elif portfolio["type"] == "pdf":

        df = read_ibrx_pdf(
            file_path
        )

    else:
        raise ValueError(
            f"Tipo de arquivo desconhecido: "
            f"{portfolio['type']}"
        )

    # validaçao
    validate_portfolio(
        df,
        portfolio["file"]
    )

    # adc info ao periodo
    df["period_start"] = pd.to_datetime(
        portfolio["start"]
    )

    df["period_end"] = pd.to_datetime(
        portfolio["end"]
    )

    df["base_date"] = pd.to_datetime(
        portfolio["base_date"]
    )

    df["index"] = "IBRX100"

    # guarda o periodo
    df["source_file"] = portfolio["file"]

    all_portfolios.append(df)


# consolidaçao
universe = pd.concat(
    all_portfolios,
    ignore_index=True
)


# ordena
universe = universe.sort_values(
    [
        "period_start",
        "ticker"
    ]
).reset_index(drop=True)


# validaçao final
print("\n" + "=" * 60)
print("VALIDAÇÃO FINAL DO UNIVERSO")
print("=" * 60)

print(
    "\nQuantidade total de registros:",
    len(universe)
)

print(
    "Quantidade de períodos:",
    universe["period_start"].nunique()
)

print(
    "Quantidade de tickers únicos no histórico:",
    universe["ticker"].nunique()
)


# cada periodo deve possuir 100 ativos
counts = (
    universe
    .groupby("period_start")["ticker"]
    .nunique()
)

print("\nAtivos por período:")
print(counts)


if not (counts == 100).all():
    raise ValueError(
        "Algum período não possui exatamente 100 ativos."
    )


# salva
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

universe.to_parquet(
    OUTPUT_PATH,
    index=False
)


# resultado
print("\n" + "=" * 60)
print("UNIVERSO IBRS-100 GERADO COM SUCESSO")
print("=" * 60)

print(
    f"\nArquivo salvo em:\n{OUTPUT_PATH}"
)

print(
    "\nLinhas salvas:",
    len(universe)
)

print(
    "Períodos:",
    universe["period_start"]
    .dt.strftime("%Y-%m-%d")
    .unique()
)

print("\nPrimeiros registros:")
print(
    universe.head(10)
)