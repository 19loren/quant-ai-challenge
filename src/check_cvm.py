import pandas as pd

path = r"C:\Users\markinkkkkj\Documents\GitHub\quant-ai-challenge\data\raw\cvm\dfp\2025\dfp_cia_aberta_DRE_con_2025.csv"

df = pd.read_csv(
    path,
    sep=";",
    encoding="latin1"
)

petro = df[
    df["DENOM_CIA"].str.contains("PETROBRAS", case=False, na=False)
]

print(
    petro[
        [
            "CD_CVM",
            "DENOM_CIA",
            "DT_FIM_EXERC",
            "CD_CONTA",
            "DS_CONTA",
            "VL_CONTA"
        ]
    ]
    .drop_duplicates()
    .to_string(index=False)
)