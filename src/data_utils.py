import pandas as pandas
import yfinance as yf

def download_single_ticker(ticker, start_date, end_date):
    
    # Baixa preços históricos de um único ativo.

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    return data
