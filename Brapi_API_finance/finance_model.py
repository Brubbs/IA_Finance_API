import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from finance_config import HORIZONTE


def dividir_cronologicamente(dados):
    datas = np.sort(dados.index.unique())
    n = len(datas)
    i1, i2 = int(n * 0.70), int(n * 0.85)

    treino = dados[dados.index.isin(datas[: i1 - HORIZONTE])]
    validacao = dados[dados.index.isin(datas[i1: i2 - HORIZONTE])]
    teste = dados[dados.index.isin(datas[i2:])]
    return treino, validacao, teste


def criar_modelo():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def avaliar(nome, modelo, features, y_real):
    X = features
    pred = modelo.predict(X)
    print(f"===== {nome} =====")
    print(confusion_matrix(y_real, pred, labels=[-1, 0, 1]))
    print(
        classification_report(
            y_real,
            pred,
            labels=[-1, 0, 1],
            target_names=["venda", "manter", "compra"],
            zero_division=0,
        )
    )
    return pred


def mostrar_importancia(modelo, features):
    print("\nImportância das features:")
    print(
        pd.Series(modelo.feature_importances_, index=features)
        .sort_values(ascending=False)
        .round(3)
    )


def mostrar_sinais(modelo, atual, features):
    if atual.empty:
        return

    ultimo = atual.groupby("ticker").tail(1)
    sinais = modelo.predict(ultimo[features])
    mapa = {1: "COMPRA", 0: "MANTER", -1: "VENDA"}
    print("\nSinal mais recente:")
    for ticker, sinal in zip(ultimo["ticker"], sinais):
        print(f"  {ticker}: {mapa[sinal]}")
