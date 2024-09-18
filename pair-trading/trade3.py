# Importa as bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# Configurações do CDI
cdi_rate = 0.13 / 252  # CDI diário aproximado (13% ao ano)

# Função para simular a pontuação ESG (valores entre 0 e 100)
def get_esg_score(ticker):
    esg_scores = {'ITUB4.SA': 80, 'SANB11.SA': 65}
    return esg_scores.get(ticker, 50)

# Função para armazenar as operações ESG em um arquivo CSV
def save_esg_operation(date, ticker, esg_score, position, cash_value, file='esg_operations.csv'):
    data = {
        'Date': [date],
        'Ticker': [ticker],
        'ESG_Score': [esg_score],
        'Position': [position],
        'Cash_Value': [cash_value]
    }
    df = pd.DataFrame(data)
    df.to_csv(file, mode='a', header=not pd.read_csv(file, nrows=1, error_bad_lines=False).empty, index=False)

# Carrega os dados de preço das ações
ticker = ['ITUB4.SA', 'SANB11.SA']
start = '2010-01-01'
end = '2024-01-01'

prices = yf.download(ticker, start=start, end=end)['Close'].dropna()

# Inicializa capital e caixa
capital = 1000
cash = capital

# Verifica as pontuações ESG
itub_esg = get_esg_score('ITUB4.SA')
sanb_esg = get_esg_score('SANB11.SA')

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

# Gerando sinais de trade baseados no z-score
sinais = pd.DataFrame(index=prices.index)
sinais['z'] = zscore(spread)
sinais['long_short_signal'] = np.select([sinais['z'] > 1, sinais['z'] < -1], [-1, 1], default=0)

# Simulação de PnL e armazenando operações ESG
for i, row in sinais.iterrows():
    itub_price = itub.loc[i]
    sanb_price = sanb.loc[i]

    # Se ESG for baixo, manter o dinheiro em caixa com CDI
    if itub_esg < 70 or sanb_esg < 70:
        cash += cash * cdi_rate
        save_esg_operation(i, 'CAIXA', min(itub_esg, sanb_esg), 'Caixa CDI', cash)
    else:
        # Calcula a posição de long/short
        sinais['position'] = sinais['long_short_signal'].diff()
        sinais['holdings_itub'] = sinais['position'].cumsum() * itub_price * capital / max(itub)
        sinais['holdings_sanb'] = -sinais['position'].cumsum() * sanb_price * capital / max(sanb)
        sinais['total_value'] = sinais['holdings_itub'] + sinais['holdings_sanb'] + cash

        # Salvar operação ESG
        save_esg_operation(i, 'ITUB4.SA', itub_esg, 'Operação Long/Short', cash)

# Visualiza o resultado acumulado do portfólio
plt.figure(figsize=(15,10))
plt.plot(sinais['total_value'], label='Valor Total do Portfólio')
plt.title('Performance do Modelo Pair Trading com CDI e ESG')
plt.legend()
plt.show()