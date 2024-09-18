import yfinance as yf
import pandas as pd

# Baixar dados das ações desde 2001
itub4 = yf.download('ITUB4.SA', start='2001-01-01', end='2023-01-01')
bbdc4 = yf.download('BBDC4.SA', start='2001-01-01', end='2023-01-01')

# Verificar dados faltantes
itub4.isnull().sum()

# Preencher ou remover dados faltantes
itub4.ffill(inplace=True)
bbdc4.ffill(inplace=True)

from statsmodels.tsa.stattools import coint

# Preços de fechamento ajustado
itub4_close = itub4['Adj Close']
bbdc4_close = bbdc4['Adj Close']

# Alinhar as duas séries temporais
aligned_data = pd.concat([itub4_close, bbdc4_close], axis=1).dropna()

# Separar as duas séries após o alinhamento
itub4_aligned = aligned_data.iloc[:, 0]  # Série 1 (ITUB4)
bbdc4_aligned = aligned_data.iloc[:, 1]  # Série 2 (BBDC4)

# Testar Cointegração
score, p_value, _ = coint(itub4_aligned, bbdc4_aligned)

if p_value < 0.05:
    print("As ações são cointegradas")
else:
    print("As ações não são cointegradas")

import numpy as np
import matplotlib.pyplot as plt

# Função para calcular o spread
def calcular_spread(preco1, preco2):
    # Verificar se as séries têm o mesmo tamanho após o alinhamento
    if len(preco1) != len(preco2):
        raise ValueError("As séries de preços não têm o mesmo comprimento.")
    
    # Calcular o beta da regressão linear entre os preços
    beta = np.polyfit(preco1, preco2, 1)[0]
    spread = preco1 - beta * preco2
    return spread

# Alinhar as duas séries temporais (remover NaNs)
aligned_data = pd.concat([itub4_close, bbdc4_close], axis=1).dropna()

# Separar as duas séries após o alinhamento
itub4_aligned = aligned_data.iloc[:, 0]
bbdc4_aligned = aligned_data.iloc[:, 1]

# Calcular o spread
spread = calcular_spread(itub4_aligned, bbdc4_aligned)

# Plotar o spread
import matplotlib.pyplot as plt
plt.plot(spread)
plt.axhline(spread.mean(), color='red')
plt.axhline(spread.mean() + 2 * spread.std(), color='green', linestyle='--')
plt.axhline(spread.mean() - 2 * spread.std(), color='green', linestyle='--')
plt.show()