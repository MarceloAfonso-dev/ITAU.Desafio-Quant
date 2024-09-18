# Importa as bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as pdr
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# Configurações de ESG
esg_threshold = 60  # Limite de pontuação ESG para filtrar empresas

# Função para simular a pontuação ESG (valores entre 0 e 100)
def get_esg_score(ticker):
    esg_scores = {'ITUB4.SA': 80, 'SANB11.SA': 65}
    return esg_scores.get(ticker, 50)

# Carrega os dados de preço das ações
ticker = ['ITUB4.SA', 'SANB11.SA']
start = '2010-01-01'
end = '2024-01-01'

prices = yf.download(ticker, start=start, end=end)['Close'].dropna()

# Verifica as pontuações ESG
itub_esg = get_esg_score('ITUB4.SA')
sanb_esg = get_esg_score('SANB11.SA')

# Se uma das empresas não atende aos requisitos ESG, o modelo não opera
if itub_esg < esg_threshold or sanb_esg < esg_threshold:
    raise ValueError(f"Uma ou mais empresas não atendem ao limite ESG. ITUB4.SA: {itub_esg}, SANB11.SA: {sanb_esg}")

# Visualização dos preços
prices.plot(figsize=(15,10), title='Preços das Ações ITUB4.SA e SANB11.SA')

# Separação das séries de preços
itub = prices['ITUB4.SA']
sanb = prices['SANB11.SA']

# Teste de Cointegração (baseado em Vidyamurthy)
score, pvalue, _ = coint(itub, sanb)
print(f'Teste de Cointegração p-valor: {pvalue}')
if pvalue > 0.05:
    raise ValueError("As séries não são cointegradas o suficiente para operar um par.")

# Regressão linear para cálculo do beta
spread_sanb = sm.add_constant(sanb)
results_reg = sm.OLS(itub, spread_sanb).fit()
beta = results_reg.params['SANB11.SA']

# Cálculo do spread
spread = itub - beta * sanb

# Função para calcular o z-score
def zscore(series):
    return (series - series.mean()) / np.std(series)

# Plota o z-score
zscore(spread).plot(figsize=(15,10), title='Z-Score do Spread')
plt.axhline(zscore(spread).mean(), color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Spread z-score', 'Média', '+1', '-1'])

# Gerando sinais de trade baseados no z-score
sinais = pd.DataFrame(index=prices.index)
sinais['z'] = zscore(spread)
sinais['long_short_signal'] = np.select([sinais['z'] > 1, sinais['z'] < -1], [-1, 1], default=0)

# Simulação de PnL baseado nos sinais gerados
capital = 1000
sinais['position'] = sinais['long_short_signal'].diff()
sinais['holdings_itub'] = sinais['position'].cumsum() * itub * capital / max(itub)
sinais['holdings_sanb'] = -sinais['position'].cumsum() * sanb * capital / max(sanb)
sinais['total_value'] = sinais['holdings_itub'] + sinais['holdings_sanb']

# Visualiza o resultado acumulado do portfólio
plt.figure(figsize=(15,10))
plt.plot(sinais['total_value'], label='Valor Total do Portfólio')
plt.title('Performance do Modelo Pair Trading')
plt.legend()
plt.show()
