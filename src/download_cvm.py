from pathlib import Path
import io
import urllib.request
import zipfile


BASE_DIR = Path(__file__).resolve().parents[1]

CVM_DIR = BASE_DIR / "data" / "raw" / "cvm" / "dfp"

# portal oficial de dados abertos da CVM (histórico desde 2010)
CVM_URL = (
    "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/"
    "dfp_cia_aberta_{year}.zip"
)


def download_year(year, force=False):
    year_dir = CVM_DIR / str(year)

    if year_dir.exists() and any(year_dir.iterdir()) and not force:
        print(f"{year}: já existe em {year_dir}, pulando")
        return

    url = CVM_URL.format(year=year)

    print(f"Baixando {year}: {url}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()

    year_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(year_dir)

    extracted = sorted(year_dir.glob("*.csv"))

    print(f"{year}: {len(extracted)} arquivos extraídos")


def main(years=range(2010, 2024), force=False):
    for year in years:
        try:
            download_year(year, force=force)
        except Exception as exc:
            print(f"{year}: falhou -> {exc}")


if __name__ == "__main__":
    main()
