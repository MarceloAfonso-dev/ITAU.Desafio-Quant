import pandas as pd
import yfinance as yf

def createFile(ticker,nome,periodo='max',caminho=r'C:\Users\55119\OneDrive\Documentos\Marcos\2024-02\Itau-asset\test\02_bases_originais\\'):
    #Buscando no Yahoo
    data = yf.Ticker(ticker)
    print(data.info)
    data_hist = data.history(period=periodo)

    #Transformando em dataframe
    df = pd.DataFrame(data_hist)

    #Exibindo
    print(f"Exibindo dados do {nome}")
    print(df)
    
    #Convertendo pra CSV
    df.to_csv(f'{caminho}{nome}.csv', index=False, sep=';', encoding='utf-8')
    print("Arquivo CSV criado com sucesso!")
    
# def criarMediaMovel(csv):
#     # df = pd.read_csv(f'{csv}')
#     # df['20_MA'] = df['Close'].rolling(window = 20).mean()
#     # df['50_MA'] = df['Close'].rolling(window = 50).mean()
#     # df.head(20)
    
# criarMediaMovel(r'C:\Users\55119\OneDrive\Documentos\Marcos\2024-02\Itau-asset\test\02_bases_originais\Café.csv')

