import numpy as np
import pandas as pd

from finance_config import HORIZONTE, LIMIAR
from finance_data import baixar, verificar


def rsi(close, n=14):
    delta = close.diff()
    ganho = delta.clip(lower=0).rolling(n).mean()
    perda = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + ganho / perda)





def criar_features(df, ticker):
    c = df["Close"]
    d = pd.DataFrame(index=df.index)
    d["ret_1d"] = c.pct_change()
    d["ret_5d"] = c.pct_change(5)
    d["ret_10d"] = c.pct_change(10)
    d["sma_ratio_9_21"] = c.rolling(9).mean() / c.rolling(21).mean()
    d["dist_sma50"] = c / c.rolling(50).mean() - 1
    d["rsi14"] = rsi(c)
    d["vol_10d"] = d["ret_1d"].rolling(10).std()
    d["vol_rel"] = df["Volume"] / df["Volume"].rolling(20).mean()
    d["amplitude"] = (df["High"] - df["Low"]) / c

    fut = c.shift(-HORIZONTE) / c - 1
    d["alvo"] = np.select(
        [fut > LIMIAR, fut < -LIMIAR],
        [1, -1],
        default=0,
    )
    d.loc[fut.isna(), "alvo"] = np.nan
    d["ticker"] = ticker
    return d


def combinar_dados(tickers):
    frames = []
    for ticker in tickers:
        df = baixar(ticker)
        verificar(df, ticker)
        frames.append(criar_features(df, ticker))
    dados = pd.concat(frames).sort_index()

    features = [col for col in dados.columns if col not in ("alvo", "ticker")]
    dados = dados.dropna(subset=features)
    atual = dados[dados["alvo"].isna()].copy()
    dados = dados.dropna(subset=["alvo"]).copy()
    dados["alvo"] = dados["alvo"].astype(int)
    return dados, atual, features
