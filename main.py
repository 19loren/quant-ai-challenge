import sys

from src import (
    backtester,
    data,
    features,
    fundamentals,
    model,
    ticker_mapping,
    universe,
    validation,
)

STEPS = [
    ("Universo IBrX100", universe.main),
    ("Preços B3", data.main),
    ("Fundamentos CVM", fundamentals.main),
    ("Validação", validation.main),
    ("Indicadores de qualidade", features.main),
    ("Pareamento ticker <-> CVM", ticker_mapping.main),
    ("Ranking multifatorial", model.main),
    ("Backtest", backtester.main),
]


def main():
    for name, step in STEPS:
        print(f"\n{'=' * 60}")
        print(f"ETAPA: {name}")
        print("=" * 60)

        try:
            step()
        except Exception as exc:
            print(f"\nFALHA na etapa '{name}': {exc}")
            sys.exit(1)

    print("\nPipeline concluído com sucesso.")


if __name__ == "__main__":
    main()
