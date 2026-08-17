from pathlib import Path

import numpy as np
import pandas as pd

from src.fundamentals import (
    CVM_DIR,
    DFP_YEARS,
    extract_account,
    normalize_cvm_code,
    normalize_dates,
    normalize_text,
    prepare_metadata,
    read_cvm_csv,
)

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "quality_features.parquet"

# contas do plano de contas CVM confirmadas como confiáveis (>99% das
# empresas) direto na base de dados real de 2024 — ver TODO.md
REVENUE_ACCOUNT = "3.01"
REVENUE_PATTERN = r"receita"

EBIT_ACCOUNT = "3.05"
EBIT_PATTERN = r"resultado"

SHORT_DEBT_ACCOUNT = "2.01.04"
LONG_DEBT_ACCOUNT = "2.02.01"
DEBT_PATTERN = r"empr[eé]stimo|financiamento|dep[oó]sito"

CASH_ACCOUNT = "1.01"
CASH_PATTERN = r"caixa"

# depreciação/amortização não tem código fixo entre empresas (vi
# 6.01.01.01 até 6.01.01.19 apontando pro mesmo conceito) — só o
# texto da conta é confiável, então soma todas as linhas do DFC que
# batem com o padrão por empresa/período
DEPRECIATION_PATTERN = r"deprecia|amortiza"


def load_statement(year, statement, required=True):
    year_dir = CVM_DIR / str(year)
    path = year_dir / f"dfp_cia_aberta_{statement}_{year}.csv"

    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Arquivo {statement} não encontrado:\n{path}"
            )
        return None

    return read_cvm_csv(path)


def extract_depreciation_amortization(dfc):
    if dfc is None or dfc.empty:
        return pd.DataFrame(
            columns=[
                "CNPJ_CIA",
                "DT_FIM_EXERC",
                "depreciacao_amortizacao",
            ]
        )

    dfc = normalize_dates(dfc, ["DT_FIM_EXERC"])

    matches = dfc[
        normalize_text(dfc["DS_CONTA"])
        .str.contains(
            DEPRECIATION_PATTERN,
            case=False,
            regex=True,
            na=False,
        )
    ].copy()

    matches["CNPJ_CIA"] = normalize_text(matches["CNPJ_CIA"])
    matches["VL_CONTA"] = pd.to_numeric(
        matches["VL_CONTA"],
        errors="coerce",
    )

    # uma empresa pode reportar D&A em mais de uma linha (ex.:
    # "depreciação do imobilizado" + "depreciação de direito de
    # uso") — soma tudo por empresa/período
    result = (
        matches
        .groupby(["CNPJ_CIA", "DT_FIM_EXERC"], as_index=False)["VL_CONTA"]
        .sum()
        .rename(columns={"VL_CONTA": "depreciacao_amortizacao"})
    )

    return result


def extract_single_account(df, account_code, column_name, description_pattern):
    rows = extract_account(
        df,
        account_code,
        description_pattern=description_pattern,
    )

    rows = rows.copy()
    rows["CNPJ_CIA"] = normalize_text(rows["CNPJ_CIA"])
    rows = normalize_dates(rows, ["DT_FIM_EXERC"])
    rows["VL_CONTA"] = pd.to_numeric(rows["VL_CONTA"], errors="coerce")

    return (
        rows[["CNPJ_CIA", "DT_FIM_EXERC", "VL_CONTA"]]
        .rename(columns={"VL_CONTA": column_name})
    )


def build_year_features(year):
    print(f"\nMontando indicadores de qualidade {year}")

    dre = load_statement(year, "DRE_con")
    bpp = load_statement(year, "BPP_con")
    bpa = load_statement(year, "BPA_con")
    dfc = load_statement(year, "DFC_MI_con", required=False)

    metadata_path = (
        CVM_DIR / str(year) / f"dfp_cia_aberta_{year}.csv"
    )
    metadata = read_cvm_csv(metadata_path)

    receita = extract_single_account(
        dre, REVENUE_ACCOUNT, "receita", REVENUE_PATTERN
    )
    ebit = extract_single_account(
        dre, EBIT_ACCOUNT, "ebit", EBIT_PATTERN
    )
    lucro_liquido = extract_single_account(
        dre, "3.11", "lucro_liquido", r"lucro|resultado"
    )
    patrimonio_liquido = extract_single_account(
        bpp, ["2.03", "2.07"], "patrimonio_liquido",
        r"patrim[oô]nio\s+l[ií]quido",
    )
    divida_curto_prazo = extract_single_account(
        bpp, SHORT_DEBT_ACCOUNT, "divida_curto_prazo", DEBT_PATTERN
    )
    divida_longo_prazo = extract_single_account(
        bpp, LONG_DEBT_ACCOUNT, "divida_longo_prazo", DEBT_PATTERN
    )
    caixa = extract_single_account(
        bpa, CASH_ACCOUNT, "caixa", CASH_PATTERN
    )
    d_e_a = extract_depreciation_amortization(dfc)

    features = receita
    for other in [
        ebit,
        lucro_liquido,
        patrimonio_liquido,
        divida_curto_prazo,
        divida_longo_prazo,
        caixa,
        d_e_a,
    ]:
        features = features.merge(
            other,
            on=["CNPJ_CIA", "DT_FIM_EXERC"],
            how="outer",
        )

    metadata = prepare_metadata(metadata)
    features = features.merge(
        metadata[["CNPJ_CIA", "CD_CVM"]].drop_duplicates(),
        on="CNPJ_CIA",
        how="left",
    )

    features["source_year"] = year

    print(f"Empresas com receita: {receita['CNPJ_CIA'].nunique():,}")
    print(f"Empresas com D&A identificado: {d_e_a['CNPJ_CIA'].nunique():,}")

    return features


def compute_ratios(df):
    df = df.copy()

    df["ebitda"] = df["ebit"] + df["depreciacao_amortizacao"].fillna(0)

    df["margem_ebitda"] = df["ebitda"] / df["receita"]
    df["margem_liquida"] = df["lucro_liquido"] / df["receita"]
    df["roe"] = df["lucro_liquido"] / df["patrimonio_liquido"]

    df["divida_liquida"] = (
        df["divida_curto_prazo"].fillna(0)
        + df["divida_longo_prazo"].fillna(0)
        - df["caixa"].fillna(0)
    )

    # alavancagem só faz sentido com ebitda positivo; empresas
    # financeiras (bancos) não têm ebitda no sentido tradicional e
    # ficam como NaN aqui — quality_score ignora componentes ausentes
    df["divida_liquida_ebitda"] = np.where(
        df["ebitda"] > 0,
        df["divida_liquida"] / df["ebitda"],
        np.nan,
    )

    ratio_columns = [
        "margem_ebitda",
        "margem_liquida",
        "roe",
        "divida_liquida_ebitda",
    ]

    df[ratio_columns] = df[ratio_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    for column in ratio_columns:
        lower = df[column].quantile(0.01)
        upper = df[column].quantile(0.99)
        df[column] = df[column].clip(lower=lower, upper=upper)

    return df


def compute_scores(df):
    df = df.copy()

    ratio_columns = [
        "margem_ebitda",
        "margem_liquida",
        "roe",
        "divida_liquida_ebitda",
    ]

    for column in ratio_columns:
        grupo_ano = df.groupby(df["DT_FIM_EXERC"].dt.year)[column]
        media = grupo_ano.transform("mean")
        desvio = grupo_ano.transform("std")
        df[f"z_{column}"] = (df[column] - media) / desvio

    df["quality_score"] = df[
        ["z_roe", "z_margem_ebitda", "z_margem_liquida"]
    ].mean(axis=1, skipna=True)

    # menor alavancagem = melhor
    df["leverage_score"] = -df["z_divida_liquida_ebitda"]

    return df


def build_quality_features(years=DFP_YEARS):
    all_years = [
        build_year_features(year)
        for year in years
    ]

    features = pd.concat(all_years, ignore_index=True)
    features = compute_ratios(features)
    features = compute_scores(features)

    return features


def save_quality_features(df):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nArquivo salvo em:\n{OUTPUT_PATH}")
    print(f"Linhas salvas: {len(df):,}")


def main():
    features = build_quality_features()

    print("\nResumo")
    print(f"Linhas: {len(features):,}")
    print(f"Empresas: {features['CNPJ_CIA'].nunique():,}")

    print("\nCobertura por indicador:")
    for column in [
        "receita",
        "ebit",
        "depreciacao_amortizacao",
        "patrimonio_liquido",
        "divida_curto_prazo",
        "quality_score",
        "leverage_score",
    ]:
        covered = features[column].notna().sum()
        print(f"  {column}: {covered:,} / {len(features):,}")

    save_quality_features(features)


if __name__ == "__main__":
    main()
