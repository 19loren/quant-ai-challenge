from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

CVM_DIR = BASE_DIR / "data" / "raw" / "cvm" / "dfp"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "fundamentals.parquet"

DFP_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

NET_INCOME_ACCOUNT = "3.11"
NET_INCOME_PATTERN = r"lucro|resultado"

# empresas "normais" reportam patrimônio líquido em 2.03; algumas
# instituições financeiras usam um plano de contas diferente onde
# 2.03 é outra coisa (ex.: "Provisões") e o PL está em 2.07 —
# por isso são dois códigos candidatos, validados pelo texto da conta
EQUITY_ACCOUNTS = ["2.03", "2.07"]
EQUITY_PATTERN = r"patrim[oô]nio\s+l[ií]quido"


def read_cvm_csv(path):
    return pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        low_memory=False,
    )


def normalize_text(series):
    return (
        series
        .astype("string")
        .str.strip()
    )


def normalize_dates(df, columns):
    df = df.copy()

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    return df


def normalize_cvm_code(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype("Int64")


def load_dfp_data(year):
    year_dir = CVM_DIR / str(year)

    required_paths = {
        "DRE_con": year_dir / f"dfp_cia_aberta_DRE_con_{year}.csv",
        "BPP_con": year_dir / f"dfp_cia_aberta_BPP_con_{year}.csv",
        "metadata": year_dir / f"dfp_cia_aberta_{year}.csv",
    }

    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo {name} não encontrado:\n{path}"
            )

    # demonstrações individuais: usadas como fallback para empresas
    # que não reportam consolidado (sem subsidiárias)
    optional_paths = {
        "DRE_ind": year_dir / f"dfp_cia_aberta_DRE_ind_{year}.csv",
        "BPP_ind": year_dir / f"dfp_cia_aberta_BPP_ind_{year}.csv",
    }

    print(f"\nCarregando DFP {year}")

    dre_con = read_cvm_csv(required_paths["DRE_con"])
    bpp_con = read_cvm_csv(required_paths["BPP_con"])
    metadata = read_cvm_csv(required_paths["metadata"])

    print(f"DRE (consolidado): {len(dre_con):,} linhas")
    print(f"BPP (consolidado): {len(bpp_con):,} linhas")
    print(f"Metadados: {len(metadata):,} linhas")

    dre_ind = None
    bpp_ind = None

    if optional_paths["DRE_ind"].exists():
        dre_ind = read_cvm_csv(optional_paths["DRE_ind"])
        print(f"DRE (individual): {len(dre_ind):,} linhas")
    else:
        print("DRE (individual): não encontrado, fallback desabilitado")

    if optional_paths["BPP_ind"].exists():
        bpp_ind = read_cvm_csv(optional_paths["BPP_ind"])
        print(f"BPP (individual): {len(bpp_ind):,} linhas")
    else:
        print("BPP (individual): não encontrado, fallback desabilitado")

    return dre_con, bpp_con, dre_ind, bpp_ind, metadata


def extract_account(df, account_code, description_pattern=None):
    """Extrai uma conta pelo CD_CONTA. O plano de contas da CVM não é
    fixo entre companhias — o mesmo código pode significar coisas
    diferentes para empresas "normais" e instituições financeiras
    (ex.: 2.03 é "Patrimônio Líquido" para ~96% das empresas, mas
    "Provisões" para outras). Quando isso é um risco, account_code
    pode ser uma lista de códigos candidatos e description_pattern
    (regex, aplicado a DS_CONTA) filtra os que não batem com o
    conceito esperado."""

    required_columns = [
        "CD_CONTA",
        "CNPJ_CIA",
        "DT_REFER",
        "DENOM_CIA",
        "CD_CVM",
        "DT_FIM_EXERC",
        "VL_CONTA",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes no arquivo CVM: "
            f"{missing}"
        )

    account_codes = (
        [account_code]
        if isinstance(account_code, str)
        else list(account_code)
    )

    result = df[
        normalize_text(df["CD_CONTA"]).isin(account_codes)
    ].copy()

    if description_pattern is not None:
        if "DS_CONTA" not in df.columns:
            raise ValueError(
                "DS_CONTA ausente — necessária para validar "
                "description_pattern."
            )

        matches_description = (
            normalize_text(result["DS_CONTA"])
            .str.contains(
                description_pattern,
                case=False,
                regex=True,
                na=False,
            )
        )

        result = result[matches_description]

        # uma empresa pode ter mais de um código candidato válido
        # (planos de contas diferentes) — mantém só um por empresa
        # e período, priorizando a ordem de account_codes
        result["_code_priority"] = (
            normalize_text(result["CD_CONTA"])
            .map({code: i for i, code in enumerate(account_codes)})
        )

        result = (
            result
            .sort_values("_code_priority")
            .drop_duplicates(
                subset=["CNPJ_CIA", "DT_FIM_EXERC"],
                keep="first",
            )
            .drop(columns="_code_priority")
        )

    return result


def prepare_metadata(metadata):
    metadata = metadata.copy()

    required_columns = [
        "CNPJ_CIA",
        "DT_REFER",
        "CD_CVM",
        "DT_RECEB",
    ]

    missing = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes nos metadados CVM: "
            f"{missing}"
        )

    metadata["CNPJ_CIA"] = normalize_text(
        metadata["CNPJ_CIA"]
    )

    metadata["CD_CVM"] = normalize_cvm_code(
        metadata["CD_CVM"]
    )

    metadata["DT_REFER"] = pd.to_datetime(
        metadata["DT_REFER"],
        errors="coerce",
    )

    metadata["DT_RECEB"] = pd.to_datetime(
        metadata["DT_RECEB"],
        errors="coerce",
    )

    if "VERSAO" in metadata.columns:
        metadata["VERSAO"] = pd.to_numeric(
            metadata["VERSAO"],
            errors="coerce",
        )
    else:
        metadata["VERSAO"] = 0

    metadata = metadata[
        [
            "CNPJ_CIA",
            "DT_REFER",
            "CD_CVM",
            "DT_RECEB",
            "VERSAO",
        ]
    ].copy()

    metadata = metadata.dropna(
        subset=[
            "CNPJ_CIA",
            "DT_REFER",
            "CD_CVM",
        ]
    )

    metadata = metadata.sort_values(
        [
            "CNPJ_CIA",
            "CD_CVM",
            "DT_REFER",
            "VERSAO",
            "DT_RECEB",
        ]
    )

    metadata = metadata.drop_duplicates(
        subset=[
            "CNPJ_CIA",
            "CD_CVM",
            "DT_REFER",
        ],
        keep="last",
    )

    metadata = metadata.drop(
        columns=["VERSAO"]
    )

    return metadata


def prepare_account_data(
    df,
    account_code,
    fundamental,
    source_year,
):
    df = df.copy()

    df = normalize_dates(
        df,
        [
            "DT_REFER",
            "DT_FIM_EXERC",
        ],
    )

    df["CNPJ_CIA"] = normalize_text(
        df["CNPJ_CIA"]
    )

    df["DENOM_CIA"] = normalize_text(
        df["DENOM_CIA"]
    )

    df["CD_CVM"] = normalize_cvm_code(
        df["CD_CVM"]
    )

    df["CD_CONTA"] = normalize_text(
        df["CD_CONTA"]
    )

    df["DS_CONTA"] = normalize_text(
        df["DS_CONTA"]
    )

    df["VL_CONTA"] = pd.to_numeric(
        df["VL_CONTA"],
        errors="coerce",
    )

    df = df[
        [
            "CNPJ_CIA",
            "DT_REFER",
            "DENOM_CIA",
            "CD_CVM",
            "DT_FIM_EXERC",
            "CD_CONTA",
            "DS_CONTA",
            "VL_CONTA",
        ]
    ].copy()

    df["fundamental"] = fundamental
    df["source_year"] = source_year

    return df


def remove_exact_duplicates(df):
    duplicate_columns = [
        "CNPJ_CIA",
        "CD_CVM",
        "DT_REFER",
        "DT_FIM_EXERC",
        "CD_CONTA",
        "fundamental",
        "VL_CONTA",
        "source_year",
    ]

    before = len(df)

    df = df.drop_duplicates(
        subset=duplicate_columns,
        keep="last",
    ).copy()

    removed = before - len(df)

    if removed > 0:
        print(
            f"Registros duplicados idênticos removidos: "
            f"{removed:,}"
        )

    return df


def extract_account_with_fallback(
    df_con,
    df_ind,
    account_code,
    source_year,
    statement_name,
    description_pattern=None,
):
    """Extrai a conta do consolidado; para empresas sem consolidado
    (sem subsidiárias), completa com a demonstração individual."""

    con_rows = extract_account(
        df_con,
        account_code,
        description_pattern=description_pattern,
    )

    if df_ind is None:
        return con_rows

    ind_rows = extract_account(
        df_ind,
        account_code,
        description_pattern=description_pattern,
    )

    con_companies = set(
        normalize_text(con_rows["CNPJ_CIA"]).dropna()
    )

    ind_only_rows = ind_rows[
        ~normalize_text(ind_rows["CNPJ_CIA"]).isin(con_companies)
    ]

    if len(ind_only_rows) > 0:
        print(
            f"{statement_name} {source_year}: "
            f"{ind_only_rows['CNPJ_CIA'].nunique()} empresa(s) sem "
            "consolidado, resolvida(s) via demonstração individual."
        )

    return pd.concat(
        [con_rows, ind_only_rows],
        ignore_index=True,
    )


def keep_latest_filing_per_period(df):
    """Quando uma empresa reapresenta o mesmo exercício (mesmo
    DT_FIM_EXERC) mais de uma vez dentro do mesmo ano-calendário da
    CVM, com DT_REFER diferentes, mantém só a apresentação mais
    recente — mesmo critério já usado em prepare_metadata()."""

    before = len(df)

    df = (
        df
        .sort_values("DT_REFER")
        .drop_duplicates(
            subset=[
                "CNPJ_CIA",
                "CD_CVM",
                "DT_FIM_EXERC",
                "CD_CONTA",
                "fundamental",
            ],
            keep="last",
        )
    )

    removed = before - len(df)

    if removed > 0:
        print(
            f"Reapresentações do mesmo exercício descartadas "
            f"(mantida a mais recente): {removed:,}"
        )

    return df


def prepare_fundamentals(
    dre_con,
    bpp_con,
    dre_ind,
    bpp_ind,
    metadata,
    source_year,
):
    dre_con = normalize_dates(
        dre_con,
        [
            "DT_REFER",
            "DT_FIM_EXERC",
        ],
    )

    bpp_con = normalize_dates(
        bpp_con,
        [
            "DT_REFER",
            "DT_FIM_EXERC",
        ],
    )

    if dre_ind is not None:
        dre_ind = normalize_dates(
            dre_ind,
            [
                "DT_REFER",
                "DT_FIM_EXERC",
            ],
        )

    if bpp_ind is not None:
        bpp_ind = normalize_dates(
            bpp_ind,
            [
                "DT_REFER",
                "DT_FIM_EXERC",
            ],
        )

    net_income = extract_account_with_fallback(
        dre_con,
        dre_ind,
        NET_INCOME_ACCOUNT,
        source_year,
        "DRE",
        description_pattern=NET_INCOME_PATTERN,
    )

    equity = extract_account_with_fallback(
        bpp_con,
        bpp_ind,
        EQUITY_ACCOUNTS,
        source_year,
        "BPP",
        description_pattern=EQUITY_PATTERN,
    )

    if net_income.empty:
        raise ValueError(
            f"Nenhum registro encontrado para "
            f"a conta {NET_INCOME_ACCOUNT} "
            f"no DFP {source_year}."
        )

    if equity.empty:
        raise ValueError(
            f"Nenhum registro encontrado para "
            f"as contas {EQUITY_ACCOUNTS} "
            f"no DFP {source_year}."
        )

    net_income = prepare_account_data(
        df=net_income,
        account_code=NET_INCOME_ACCOUNT,
        fundamental="net_income",
        source_year=source_year,
    )

    equity = prepare_account_data(
        df=equity,
        account_code=EQUITY_ACCOUNTS,
        fundamental="equity",
        source_year=source_year,
    )

    fundamentals = pd.concat(
        [
            net_income,
            equity,
        ],
        ignore_index=True,
    )

    fundamentals = remove_exact_duplicates(
        fundamentals
    )

    fundamentals = keep_latest_filing_per_period(
        fundamentals
    )

    metadata = prepare_metadata(
        metadata
    )

    fundamentals = fundamentals.merge(
        metadata,
        on=[
            "CNPJ_CIA",
            "CD_CVM",
            "DT_REFER",
        ],
        how="left",
        validate="many_to_one",
    )

    return fundamentals


def normalize_fundamentals(df):
    df = df.copy()

    df["CNPJ_CIA"] = normalize_text(
        df["CNPJ_CIA"]
    )

    df["DENOM_CIA"] = normalize_text(
        df["DENOM_CIA"]
    )

    df["CD_CONTA"] = normalize_text(
        df["CD_CONTA"]
    )

    df["DS_CONTA"] = normalize_text(
        df["DS_CONTA"]
    )

    df["fundamental"] = normalize_text(
        df["fundamental"]
    )

    df["CD_CVM"] = normalize_cvm_code(
        df["CD_CVM"]
    )

    df["DT_REFER"] = pd.to_datetime(
        df["DT_REFER"],
        errors="coerce",
    )

    df["DT_FIM_EXERC"] = pd.to_datetime(
        df["DT_FIM_EXERC"],
        errors="coerce",
    )

    df["DT_RECEB"] = pd.to_datetime(
        df["DT_RECEB"],
        errors="coerce",
    )

    df["VL_CONTA"] = pd.to_numeric(
        df["VL_CONTA"],
        errors="coerce",
    )

    df["source_year"] = pd.to_numeric(
        df["source_year"],
        errors="coerce",
    ).astype("Int64")

    return df


def validate_fundamentals(df):
    required_columns = [
        "CNPJ_CIA",
        "DT_REFER",
        "DENOM_CIA",
        "CD_CVM",
        "DT_FIM_EXERC",
        "CD_CONTA",
        "DS_CONTA",
        "VL_CONTA",
        "fundamental",
        "DT_RECEB",
        "source_year",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            f"{missing_columns}"
        )

    critical_columns = [
        "CNPJ_CIA",
        "DT_REFER",
        "CD_CVM",
        "DT_FIM_EXERC",
        "CD_CONTA",
        "VL_CONTA",
        "fundamental",
        "source_year",
    ]

    null_counts = (
        df[critical_columns]
        .isna()
        .sum()
    )

    null_counts = null_counts[
        null_counts > 0
    ]

    if not null_counts.empty:
        print("\nValores ausentes encontrados:")
        print(
            null_counts.to_string()
        )

        raise ValueError(
            "Existem valores ausentes em "
            "campos críticos dos fundamentos."
        )

    valid_fundamentals = {
        "net_income",
        "equity",
    }

    actual_fundamentals = set(
        df["fundamental"].unique()
    )

    unexpected_fundamentals = (
        actual_fundamentals
        - valid_fundamentals
    )

    if unexpected_fundamentals:
        raise ValueError(
            "Tipos de fundamentos inesperados: "
            f"{unexpected_fundamentals}"
        )

    actual_years = set(
        df["source_year"]
        .dropna()
        .astype(int)
        .unique()
    )

    unexpected_years = (
        actual_years
        - set(DFP_YEARS)
    )

    if unexpected_years:
        raise ValueError(
            "Anos de origem inesperados: "
            f"{unexpected_years}"
        )

    duplicate_columns = [
        "CNPJ_CIA",
        "CD_CVM",
        "DT_REFER",
        "DT_FIM_EXERC",
        "CD_CONTA",
        "fundamental",
        "VL_CONTA",
        "source_year",
    ]

    duplicate_mask = df.duplicated(
        subset=duplicate_columns,
        keep=False,
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:
        print(
            "\nRegistros envolvidos em duplicidades:"
        )

        print(
            df.loc[
                duplicate_mask
            ]
            .sort_values(
                [
                    "CD_CVM",
                    "DT_REFER",
                    "DT_FIM_EXERC",
                    "fundamental",
                ]
            )
            .head(30)
            .to_string(index=False)
        )

        raise ValueError(
            f"Existem {duplicate_count:,} linhas "
            "envolvidas em duplicidades."
        )

    if df.empty:
        raise ValueError(
            "O DataFrame final de fundamentos está vazio."
        )

    print("\nValidação dos fundamentos")
    print(
        f"Dimensões: {df.shape}"
    )

    print(
        f"Empresas: "
        f"{df['CD_CVM'].nunique():,}"
    )

    print(
        f"Tickers não disponíveis nesta etapa: "
        "a identificação será feita posteriormente."
    )

    print(
        f"Tipos de fundamentos: "
        f"{df['fundamental'].nunique()}"
    )

    print(
        f"Duplicidades: {duplicate_count}"
    )

    print("\nRegistros por tipo:")
    print(
        df["fundamental"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nRegistros por ano de origem:")
    print(
        df["source_year"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nValidação dos fundamentos concluída.")


def process_all_years():
    all_fundamentals = []

    for year in DFP_YEARS:
        dre_con, bpp_con, dre_ind, bpp_ind, metadata = load_dfp_data(
            year
        )

        fundamentals = prepare_fundamentals(
            dre_con=dre_con,
            bpp_con=bpp_con,
            dre_ind=dre_ind,
            bpp_ind=bpp_ind,
            metadata=metadata,
            source_year=year,
        )

        fundamentals = normalize_fundamentals(
            fundamentals
        )

        all_fundamentals.append(
            fundamentals
        )

    if not all_fundamentals:
        raise ValueError(
            "Nenhum fundamento foi processado."
        )

    fundamentals = pd.concat(
        all_fundamentals,
        ignore_index=True,
    )

    fundamentals = normalize_fundamentals(
        fundamentals
    )

    return fundamentals


def save_fundamentals(df):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )


def main():
    print(
        "Iniciando processamento dos fundamentos CVM"
    )

    print(
        f"Diretório CVM: {CVM_DIR}"
    )

    print(
        f"Anos: {DFP_YEARS}"
    )

    fundamentals = process_all_years()

    validate_fundamentals(
        fundamentals
    )

    print("\nResumo final")

    print(
        f"Linhas: {len(fundamentals):,}"
    )

    print(
        f"Empresas: "
        f"{fundamentals['CD_CVM'].nunique():,}"
    )

    print(
        f"Período de referência: "
        f"{fundamentals['DT_REFER'].min().date()} "
        f"-> "
        f"{fundamentals['DT_REFER'].max().date()}"
    )

    print(
        f"Período contábil: "
        f"{fundamentals['DT_FIM_EXERC'].min().date()} "
        f"-> "
        f"{fundamentals['DT_FIM_EXERC'].max().date()}"
    )

    valid_receipt_dates = fundamentals[
        "DT_RECEB"
    ].dropna()

    if not valid_receipt_dates.empty:
        print(
            f"Período de recebimento: "
            f"{valid_receipt_dates.min().date()} "
            f"-> "
            f"{valid_receipt_dates.max().date()}"
        )

    save_fundamentals(
        fundamentals
    )

    print(
        f"\nArquivo salvo em:\n{OUTPUT_PATH}"
    )

    print(
        f"Linhas salvas: "
        f"{len(fundamentals):,}"
    )


if __name__ == "__main__":
    main()