import pandas as pd
import yfinance as yf


def baixar(ticker):
    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if df.empty:
        raise ValueError(
            f"Nenhum dado diário válido foi retornado para {ticker}. "
            "Verifique o ticker e a conexão com o Yahoo Finance."
        )
    return df


def verificar(df, ticker):
    """Checagem básica de qualidade dos dados (informativa, não altera o DataFrame)."""
    print(f"--- Verificação: {ticker} ---")
    if df.empty:
        print("Período: sem dados disponíveis\n")
        return

    print(
        "Período:",
        df.index.min().date(),
        "->",
        df.index.max().date(),
        "| linhas:",
        len(df),
    )
    print("Nulos:", int(df.isna().sum().sum()))
    print(
        "Preços <= 0:",
        int((df[["Open", "High", "Low", "Close"]] <= 0).sum().sum()),
    )
    print("Volume zero:", int((df["Volume"] == 0).sum()))
    print("Datas duplicadas:", int(df.index.duplicated().sum()))

    ohlc_ruim = (
        (df["High"] < df["Low"])
        | (df["High"] < df["Close"])
        | (df["Low"] > df["Close"])
    )
    print("OHLC inconsistente:", int(ohlc_ruim.sum()))

    ret = df["Close"].pct_change()
    saltos = ret[ret.abs() > 0.15]
    print("Retornos diários > 15%:", len(saltos))
    if len(saltos):
        print(saltos.round(3).to_string())

    gaps = df.index.to_series().diff().dt.days
    maior_intervalo = gaps.max()
    if pd.isna(maior_intervalo):
        print("Maior intervalo entre pregões (dias): não disponível\n")
    else:
        print("Maior intervalo entre pregões (dias):", int(maior_intervalo), "\n")
