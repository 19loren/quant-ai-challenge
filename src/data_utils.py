import pandas as pandas
import yfinance as yf
import numpy as np

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

def download_multiple_tickers(tickers, start_date, end_date):
    # Baixa preços históricos de múltiplos ativos.
    prices = {}

    for ticker in tickers:

        print(f"Baixando {ticker}...")

        try:

            df = download_single_ticker(
                ticker,
                start_date,
                end_date
            )

            if not df.empty:
                prices[ticker] = df

        except Exception as e:

            print(f"Erro em {ticker}: {e}")

    return prices

def prices_dictionary_to_dataframe(prices_dict):
    
    #Converte o dicionário de DataFrames em um único DataFrame.

    dfs = []

    for ticker, df in prices_dict.items():

        temp = df.copy()

        temp = temp.reset_index()

        temp["ticker"] = ticker

        dfs.append(temp)

    prices = pd.concat(
        dfs,
        ignore_index=True
    )

    return prices
