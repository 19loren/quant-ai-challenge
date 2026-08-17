import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.fundamentals import CVM_DIR, DFP_YEARS, read_cvm_csv

BASE_DIR = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = BASE_DIR / "data" / "processed" / "universe.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "ticker_cvm_map.parquet"

CORPORATE_SUFFIXES = (
    r"\b(SA|S A|HOLDING|HOLDINGS|PARTICIPACOES|PART|CIA|COMPANHIA|"
    r"GRUPO|BRASIL|BR|ON|PN|N1|N2|NM|EDJ)\b"
)

# tickers sem match automático (nome no boletim IBrX é uma sigla
# curta ou apelido, não bate por normalização/substring) — CD_CVM
# conferido linha a linha contra data/raw/cvm/dfp/*/dfp_cia_aberta_
# {ano}.csv, nunca de memória. Ver TODO.md para o que ficou de fora
# por falta de evidência suficiente pra desambiguar.
MANUAL_OVERRIDES = {
    "B3SA3": 21610,    # B3 S.A. - BRASIL, BOLSA, BALCÃO
    "BBDC3": 906,      # BCO BRADESCO S.A. (não a Bradesco Leasing)
    "BBDC4": 906,
    "BBSE3": 23159,    # BB SEGURIDADE PARTICIPAÇÕES S.A.
    "BPAC11": 22616,   # BCO BTG PACTUAL S.A.
    "CEAB3": 24848,    # C&A MODAS S.A.
    "CMIG4": 2453,     # CIA ENERGETICA DE MINAS GERAIS - CEMIG (holding, não Distribuição/Geração)
    "CPLE3": 14311,    # CIA PARANAENSE DE ENERGIA - COPEL (holding)
    "CPLE5": 14311,
    "CPLE6": 14311,
    "CRFB3": 14826,    # CIA BRASILEIRA DE DISTRIBUICAO (Carrefour Brasil)
    "CSNA3": 4030,     # CIA SIDERURGICA NACIONAL
    "CVCB3": 23310,    # CVC BRASIL OPERADORA E AGÊNCIA DE VIAGENS S.A.
    "CYRE3": 14460,    # CYRELA BRAZIL REALTY S.A.
    "ECOR3": 19453,    # ECORODOVIAS INFRAESTRUTURA E LOGÍSTICA S.A. (holding)
    "EZTC3": 20770,    # EZ TEC EMPREEND. E PARTICIPACOES S.A.
    "GGPS3": 25712,    # GPS PARTICIPAÇÕES E EMPREENDIMENTOS S.A.
    "IRBR3": 24180,    # IRB - BRASIL RESSEGUROS S.A.
    "ITUB3": 19348,    # ITAU UNIBANCO HOLDING S.A.
    "ITUB4": 19348,
    "MGLU3": 22470,    # MAGAZINE LUIZA S.A.
    "MRVE3": 20915,    # MRV ENGENHARIA E PARTICIPACOES S.A.
    "NATU3": 19550,    # NATURA COSMÉTICOS S.A. (ticker anterior à fusão)
    "NTCO3": 24783,    # NATURA &CO HOLDING S.A. (ticker atual, pós-fusão Avon)
    "RADL3": 5258,     # RAIA DROGASIL S.A.
    "RENT3": 19739,    # LOCALIZA RENT A CAR S.A. (não a Localiza Fleet)
    "SANB11": 20532,   # BCO SANTANDER (BRASIL) S.A. (não a Santander Leasing)
    "SBSP3": 14443,    # CIA SANEAMENTO BASICO EST SAO PAULO (SABESP)
    "SMFT3": 24260,    # SMARTFIT ESCOLA DE GINÁSTICA E DANÇA S.A.
}


def strip_accents(text):
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize_name(name):
    if pd.isna(name):
        return ""

    name = strip_accents(str(name)).upper().strip()
    name = name.replace("S/A", "SA").replace("S.A.", "SA").replace("S.A", "SA")
    name = re.sub(r"[^A-Z0-9 ]", " ", name)
    name = re.sub(CORPORATE_SUFFIXES, " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def build_cvm_registry():
    """Uma linha por (CNPJ_CIA, CD_CVM) com o nome mais recente
    visto em qualquer ano do DFP baixado."""

    all_metadata = []

    for year in DFP_YEARS:
        path = CVM_DIR / str(year) / f"dfp_cia_aberta_{year}.csv"

        if not path.exists():
            continue

        metadata = read_cvm_csv(path)
        metadata = metadata[["CNPJ_CIA", "DENOM_CIA", "CD_CVM", "DT_REFER"]]
        all_metadata.append(metadata)

    registry = pd.concat(all_metadata, ignore_index=True)
    registry["DT_REFER"] = pd.to_datetime(
        registry["DT_REFER"], errors="coerce"
    )

    registry = (
        registry
        .sort_values("DT_REFER")
        .drop_duplicates(subset=["CNPJ_CIA", "CD_CVM"], keep="last")
    )

    registry["nome_normalizado"] = registry["DENOM_CIA"].apply(normalize_name)

    return registry[
        ["CNPJ_CIA", "CD_CVM", "DENOM_CIA", "nome_normalizado"]
    ]


def match_by_normalized_name(tickers, registry):
    return tickers.merge(
        registry,
        on="nome_normalizado",
        how="inner",
    )


MIN_MATCH_LENGTH = 4


def match_by_substring(tickers, registry):
    """Para os que sobraram: nome do ticker contido no razão social
    CVM ou vice-versa (ex.: "ELETROBRAS" está contido em
    "CENTRAIS ELETRICAS BRASILEIRAS ELETROBRAS"). Nomes muito curtos
    após a normalização (sufixos societários removidos podem deixar
    só 1-2 letras) são ignorados — senão viram substring de quase
    tudo e geram matches falsos."""

    candidates = []

    for _, row in tickers.iterrows():
        nome = row["nome_normalizado"]

        if len(nome) < MIN_MATCH_LENGTH:
            continue

        mask = registry["nome_normalizado"].apply(
            lambda cvm_nome: (
                len(cvm_nome) >= MIN_MATCH_LENGTH
                and (nome in cvm_nome or cvm_nome in nome)
            )
        )

        matched = registry[mask]

        if len(matched) == 1:
            candidates.append(
                {
                    "ticker": row["ticker"],
                    "name": row["name"],
                    **matched.iloc[0].to_dict(),
                }
            )

    return pd.DataFrame(candidates)


def build_ticker_cvm_map():
    universe = pd.read_parquet(UNIVERSE_PATH)
    registry = build_cvm_registry()

    tickers = (
        universe[["ticker", "name"]]
        .drop_duplicates("ticker")
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    tickers["nome_normalizado"] = tickers["name"].apply(normalize_name)

    exact = match_by_normalized_name(tickers, registry)
    exact = exact.drop_duplicates(subset="ticker", keep="first")

    matched_tickers = set(exact["ticker"])
    remaining = tickers[~tickers["ticker"].isin(matched_tickers)]

    substring = match_by_substring(remaining, registry)

    matched_tickers |= set(substring["ticker"]) if not substring.empty else set()
    remaining = tickers[~tickers["ticker"].isin(matched_tickers)]

    override_rows = []
    for _, row in remaining.iterrows():
        cd_cvm = MANUAL_OVERRIDES.get(row["ticker"])
        if cd_cvm is None:
            continue

        registry_row = registry[registry["CD_CVM"] == cd_cvm]
        if registry_row.empty:
            continue

        override_rows.append(
            {
                "ticker": row["ticker"],
                "name": row["name"],
                **registry_row.iloc[0].to_dict(),
            }
        )

    overrides = pd.DataFrame(override_rows)

    result = pd.concat(
        [exact, substring, overrides],
        ignore_index=True,
    )
    result = result.drop_duplicates(subset="ticker", keep="first")

    unmatched = tickers[~tickers["ticker"].isin(result["ticker"])]

    return result[
        ["ticker", "name", "CNPJ_CIA", "CD_CVM", "DENOM_CIA"]
    ].sort_values("ticker").reset_index(drop=True), unmatched


def save_ticker_cvm_map(mapping):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nArquivo salvo em:\n{OUTPUT_PATH}")
    print(f"Linhas salvas: {len(mapping):,}")


def main():
    mapping, unmatched = build_ticker_cvm_map()

    total = mapping["ticker"].nunique() + len(unmatched)

    print(f"Tickers no universo: {total}")
    print(f"Tickers com CD_CVM: {mapping['ticker'].nunique()}")
    print(f"Tickers sem correspondência: {len(unmatched)}")

    duplicated_cvm = mapping["CD_CVM"].duplicated(keep=False)
    if duplicated_cvm.any():
        print("\nCD_CVM usado por mais de um ticker (classes distintas — ok):")
        print(
            mapping[duplicated_cvm][["ticker", "CD_CVM", "DENOM_CIA"]]
            .sort_values("CD_CVM")
            .to_string(index=False)
        )

    if not unmatched.empty:
        print("\nSem correspondência:")
        print(unmatched[["ticker", "name"]].to_string(index=False))

    save_ticker_cvm_map(mapping)


if __name__ == "__main__":
    main()
