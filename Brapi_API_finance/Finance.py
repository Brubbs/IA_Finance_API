"""
Pipeline simples: yfinance -> features -> rótulo -> Random Forest -> avaliação.

Classes: 1 = compra, 0 = manutenção, -1 = venda.
pip install yfinance pandas numpy scikit-learn
"""
import numpy as np

from finance_config import TICKERS
from finance_features import combinar_dados
from finance_model import (
    criar_modelo,
    dividir_cronologicamente,
    avaliar,
    mostrar_importancia,
    mostrar_sinais,
)


def main():
    dados, atual, features = combinar_dados(TICKERS)
    treino, validacao, teste = dividir_cronologicamente(dados)

    X_treino = treino[features]
    y_treino = treino["alvo"]
    X_validacao = validacao[features]
    y_validacao = validacao["alvo"]
    X_teste = teste[features]
    y_teste = teste["alvo"]

    print("Distribuição das classes (treino):")
    print(y_treino.value_counts(normalize=True).round(3), "\n")

    modelo = criar_modelo()
    modelo.fit(X_treino, y_treino)

    avaliar("VALIDAÇÃO", modelo, X_validacao, y_validacao)
    avaliar("TESTE", modelo, X_teste, y_teste)

    baseline = np.full(len(y_teste), y_treino.mode()[0])
    print(
        "Acurácia baseline (classe mais comum):",
        (baseline == y_teste).mean().round(3),
    )
    print(
        "Acurácia do modelo no teste:          ",
        (modelo.predict(X_teste) == y_teste).mean().round(3),
    )

    mostrar_importancia(modelo, features)
    mostrar_sinais(modelo, atual, features)


if __name__ == "__main__":
    main()