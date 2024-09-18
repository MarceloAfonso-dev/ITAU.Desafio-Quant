# Importa as bibliotecas
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pandas_datareader import data as pdr
!pip install yfinance
import yfinance as yf

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

ticker = ['ITUB4.SA','SANB11.SA']
start = '2010-01-01'
end = '2024-01-01'

prices = yf.download(ticker, start=start, end=end)['Close'].dropna()

plt.style.use('seaborn')

#Verifica o comportamento das suas séries
prices.plot(figsize=(15,10))

# Salva a série de preços e dois objetos diferentes

itub = prices[['ITUB4.SA']]
sanb = prices[['SANB11.SA']]

# Realizar o teste de cointegração

score, pvalue, _ = coint(itub, sanb, maxlag = 1)
print(' Teste p-valor da Cointegração ' + str(pvalue))

# Extraia as séries de preços do DataFrame
sanb = prices['SANB11.SA']
itub = prices['ITUB4.SA']

# Adicione uma constante à série sanb para a regressão
spread_sanb = sm.add_constant(sanb)

# Realize a regressão linear
results_reg = sm.OLS(itub, spread_sanb).fit()

# Calcule o beta da regressão
b = results_reg.params['SANB11.SA']

# Calcule o spread
spread = itub - b * sanb

# Exiba o resultado
print(spread.head())

spread

# Cria a função para calcular o z-score

def zscore(series):
  return (series - series.mean()) / np.std(series)

# Plota o z-score

zscore(spread).plot(figsize=(15,10))
plt.axhline(zscore(spread).mean(), color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Spread z-score', 'Mean', '+1', '-1']);

# Cria um df para os sinais de trade
sinais = pd.DataFrame()
sinais['itub'] = itub
sinais['sanb'] = sanb
ratios = sinais.sanb / sinais.itub

# Calcula o z-score e define os limites superiores e inferiores
sinais['z'] = zscore(ratios)
sinais['z_limite_superior'] = np.mean(sinais['z']) + np.std(sinais['z'])
sinais['z_limite_inferior'] = np.mean(sinais['z']) - np.std(sinais['z'])

# Cria o sinal - vendido se o z-score é maior que o limite superior, senão comprado
sinais['sinais1'] = 0
sinais['sinais1'] = np.select([sinais['z'] > \
                                  sinais['z_limite_superior'], sinais['z'] < sinais['z_limite_inferior']], [-1,1], default=0)

# Diferencia em primeira order para obter a posição da ação
sinais['positions1'] = sinais['sinais1'].diff()
sinais['sinais2'] = -sinais['sinais1']
sinais['positions2'] = sinais['sinais2'].diff()

# Capital investido para obter PnL
capital = 1000

# Shares to buy for each position
positions1 = capital // max(sinais['sanb'])
positions2 = capital // max(sinais['itub'])

# Cria o DataFrame do portfólio
portifolio = pd.DataFrame()

# Cálculo para 'sanb'
portifolio['sanb'] = sinais['sanb']
portifolio['holding_sanb'] = sinais['positions1'].cumsum() * sinais['sanb'] * positions1
portifolio['cash_sanb'] = capital - (sinais['positions1'] * sinais['sanb'] * positions1).cumsum()
portifolio['total_sanb'] = portifolio['holding_sanb'] + portifolio['cash_sanb']
portifolio['return_sanb'] = portifolio['total_sanb'].pct_change()

# Cálculo para 'itub'
portifolio['itub'] = sinais['itub']
portifolio['holding_itub'] = sinais['positions2'].cumsum() * sinais['itub'] * positions2
portifolio['cash_itub'] = capital - (sinais['positions2'] * sinais['itub'] * positions2).cumsum()
portifolio['total_itub'] = portifolio['holding_itub'] + portifolio['cash_itub']
portifolio['return_itub'] = portifolio['total_itub'].pct_change()

# Adiciona as posições no portfólio
portifolio['positions1'] = sinais['positions1']
portifolio['positions2'] = sinais['positions2']

# Calcula o total do portfólio somando ambos os ativos
portifolio['total'] = portifolio['total_sanb'] + portifolio['total_itub']

# Remove valores nulos
portifolio = portifolio.dropna()

plt.figure(figsize=(15,10))
plt.plot(portifolio['total'])