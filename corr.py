#create a correlation matrix for testing the top 10 stocks of the S&P 500
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'META', 'GOOGL', 'TSLA', 'BRK-B', 'GOOG', 'AVGO']

#Obtain historical data
data = yf.download(tickers, start="2019-01-01", end="2024-11-20")['Adj Close']

#Calculate daily returns

returns = data.pct_change().dropna()

#compute correlation matrix 
correlation_matrix = returns.corr()

#display as a heatmap 
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.show()