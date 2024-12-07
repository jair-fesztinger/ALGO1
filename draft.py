import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as stat

#specify parameters
tickers = ['AAPL','MSFT']
start = '2019-12-31'
end = '2024-11-09' 

#retrieve data 
data = pd.DataFrame()
returns = pd.DataFrame()

for stock in tickers:
    prices = yf.download(stock, start, end)

    # Convert the index to just the date, removing time component
    prices.index = prices.index.date
    data[stock] = prices['Close']

# Reset the index to make 'date' a column in the DataFrame
data = data.reset_index()
data = data.rename(columns={'index': 'time'})  # Rename index to 'time' 

#compute daily percentage returns
returns = data[['time', 'AAPL', 'MSFT']].set_index('time').pct_change().dropna()
returns = returns.reset_index()

#visual display of returns
fig1 = px.line(returns, x='time', y=['AAPL', 'MSFT'], title= 'Daily Returns - AAPL vs MSFT')
fig1.show()



#confirm if they are correlated by applying Ordinary Least Squares and ADF Test
OLS_Method = stat.OLS(returns['AAPL'], stat.add_constant(returns['MSFT'])).fit()
Beta = OLS_Method.params[0]

#calculate the spread with Beta
returns['spread'] = returns['AAPL'] - Beta * returns['MSFT']

#visualize the spread 
fig2 = px.line(returns, x = returns.index, y='spread', title= 'Spread between Microsoft anf Apple (Returns)')
fig2.show()

ADF_Test = adfuller(OLS_Method.resid)

#logic test
if ADF_Test[0] <= ADF_Test[4]['10%'] and ADF_Test[1] <= .1:
    print("Pair of securities is co-integrated")
else:
    print("Pair of securities is not co-integrated")


#Bollinger Bands 
#calculating the moving average and the standard deviation 
def bollinger_bands(data, spread_cl='spread', window=15, sd=1.8):
    data['SMA'] = data[spread_cl].rolling(window=window).mean()
    data['Upper'] = data['SMA'] + (sd * data[spread_cl].rolling(window=window).std())
    data['Lower'] = data['SMA'] - (sd * data[spread_cl].rolling(window=window).std())

    return data

#Apply bollinger bands to the spread
returns = bollinger_bands(returns, spread_cl='spread', window=15, sd=1.8)


#impement an RSI to add further confidence in our entries/exits
def compute_rsi(returns, column):
   #price changes
   delta = returns[column].diff() 

   #seperate gains and losses
   gain = delta.where(delta>0, 0)
   loss = -delta.where(delta<0, 0) #the negative serves as my absolute value

   #compute rolling average
   avg_gain = gain.rolling(14).mean()
   avg_loss = loss.rolling(14).mean()

   #calculate RSI
   RSI = 100*avg_gain/(avg_loss+avg_gain)

   #add RSI column to the dataframe 
   returns['Spread_RSI'] = RSI
   return returns

#Apply function
returns = compute_rsi(returns, column='spread')

#plot RSI
fig_rsi = px.line(returns, x='time', y='Spread_RSI', title='RSI of Spread', labels={'x':'Time', 'y':'RSI'})

#Include overbought (67) and oversold (33) lines
fig_rsi.add_scatter(x=returns['time'], y=[65]*len(returns), mode='lines', name='Overbought (65)', line=dict(dash='dash', color='red'))
fig_rsi.add_scatter(x=returns['time'], y=[35]*len(returns), mode='lines', name='Oversold (35)', line=dict(dash='dash', color='green'))

fig_rsi.show()

#Generate Trade Signal 
def trade_strategy(data, spread_cl='spread', upper_cl='Upper', lower_cl='Lower', spread_RSI_cl='Spread_RSI', threshold=35):
    #Initialize columns to store trade signals
    data['Signal']=0
    data['Buy']=np.nan
    data['Sell']=np.nan

    for i in range(1, len(data)):
        #Buy signal: Spread goes below the Lower Band
        if (data[spread_cl].iloc[i-1] > data[lower_cl].iloc[i-1] and data[spread_cl].iloc[i] < data[lower_cl].iloc[i] and
            data[spread_cl].iloc[i] < data[lower_cl].iloc[i] and data[spread_RSI_cl].iloc[i] < threshold):
            
            data['Signal'].iloc[i] = 1 #Buy signal 
            data['Buy'].iloc[i] = data[spread_cl].iloc[i] #Mark buy price 

        #Sell signal: Spread goes over the Upper Band
        elif (data[spread_cl].iloc[i-1] < data[upper_cl].iloc[i-1] and data[spread_cl].iloc[i] > data[upper_cl].iloc[i] and
            data[spread_cl].iloc[i] > data[upper_cl].iloc[i] and data[spread_RSI_cl].iloc[i] > (100-threshold)):
            
            
            data['Signal'].iloc[i]= -1 #Sell signal 
            data['Sell'].iloc[i]=data[spread_cl].iloc[i] #Mark sell price 

    return data

returns = trade_strategy(returns, spread_cl='spread', upper_cl='Upper', lower_cl='Lower')

#plot the spread with trade signals
fig = px.line(returns, x=returns.index, y='spread', title='Spread with Bollinger Bands and Trade Signals')
fig.add_scatter(x=returns.index, y=returns['Upper'], mode='lines', name='Upper Band')
fig.add_scatter(x=returns.index, y=returns['Lower'], mode='lines', name='Lower Band')
fig.add_scatter(x=returns.index, y=returns['SMA'], mode='lines', name='Mean')
fig.add_scatter(x=returns.index, y=returns['Buy'], mode='markers', name='Buy Signal', marker=dict(color='green', size=10))
fig.add_scatter(x=returns.index, y=returns['Sell'], mode='markers', name='Sell Signal', marker=dict(color='red', size=10))
fig.show()

#Implement long and short strategy 
#Recall Spread = AAPL - BETA * MSFT

#Signal 1 implies long in Lower Band -> Short MSFT, Long AAPL
#Signal -1 implies long in Upper Band -> Short AAPL, Long MSFT 
position = 0 # 0 = no position, 1 = long spread, -1 = short spread
long = 0
short = 0
cumulative_total = 0
transaction_cost_rate = 0
shares_per_position = 100

pnl = []
cumulative_pnl=[]
position_log = [] 


#Apply signals and generate PnL (backtest)
for i in range(len(returns)):
    signal = returns['Signal'].iloc[i]
    spread = returns['spread'].iloc[i]
    upper_band = returns['Upper'].iloc[i]
    lower_band = returns['Lower'].iloc[i]
    mean_band = returns['SMA'].iloc[i]


    #open a long position: signal 1 aka spread below lower band
    if signal == 1 and position == 0:
        entry_price_AAPL = data['AAPL'].iloc[i]
        entry_price_MSFT = data['MSFT'].iloc[i]

        # Calculate transaction costs for entry
        transaction_cost = transaction_cost_rate * (entry_price_AAPL + entry_price_MSFT) * shares_per_position

        # Log trade details
        position_log.append({
            'Time': data.index[i],
            'Signal': 'Long Spread',
            'Long Asset': 'AAPL',
            'Short Asset': 'MSFT',
            'Entry Price Long': entry_price_AAPL,
            'Entry Price Short': entry_price_MSFT
        })

        # Go long AAPL and short MSFT
        position = 1
        long = 1
        short = 1


    #Close a long position: Signal = -1 or neutral
    elif position == 1 and spread >= upper_band:
        exit_price_AAPL = data['AAPL'].iloc[i]
        exit_price_MSFT = data['MSFT'].iloc[i]

        # Calculate transaction costs for exit
        transaction_cost = transaction_cost_rate * (exit_price_AAPL + exit_price_MSFT) * shares_per_position
        
        pnl_trade = ((exit_price_AAPL-entry_price_AAPL) * shares_per_position - (exit_price_MSFT-entry_price_MSFT) * shares_per_position) - 2 * transaction_cost
        pnl.append(pnl_trade)

        cumulative_total += pnl_trade
        cumulative_pnl.append(cumulative_total)

        # Log trade details
        position_log.append({
            'Time': data.index[i],
            'Signal': 'Close Long Spread',
            'Long Asset': 'AAPL',
            'Short Asset': 'MSFT',
            'Exit Price Long': exit_price_AAPL,
            'Exit Price Short': exit_price_MSFT,
            'PnL': pnl_trade 
        })
        
        #reset position
        position = 0
        long = 0
        short = 0
    
    #Open a short position: Signal = -1 aka above the upper band
    elif signal == -1 and position == 0:
        entry_price_AAPL = data['AAPL'].iloc[i]
        entry_price_MSFT = data['MSFT'].iloc[i]

        # Calculate transaction costs for entry
        transaction_cost = transaction_cost_rate * (entry_price_AAPL + entry_price_MSFT) * shares_per_position
        
        # Log trade details
        position_log.append({
            'Time': data.index[i],
            'Signal': 'Short Spread',
            'Long Asset': 'MSFT',
            'Short Asset': 'AAPL',
            'Entry Price Long': entry_price_MSFT,
            'Entry Price Short': entry_price_AAPL
        })
        
        # Go long MSFT and short AAPL
        position = -1
        long = 1
        short = 1


    #Close a short position: Signal = 1 or neutral
    elif position == -1 and spread <= lower_band:
        exit_price_AAPL = data['AAPL'].iloc[i]
        exit_price_MSFT = data['MSFT'].iloc[i]

        # Calculate transaction costs for exit
        transaction_cost = transaction_cost_rate * (exit_price_AAPL + exit_price_MSFT) * shares_per_position
        
        pnl_trade = ((exit_price_MSFT-entry_price_MSFT) * shares_per_position - (exit_price_AAPL-entry_price_AAPL) * shares_per_position) - 2 * transaction_cost
        pnl.append(pnl_trade)
        
        cumulative_total += pnl_trade
        cumulative_pnl.append(cumulative_total)

        # Log trade details
        position_log.append({
            'Time': data.index[i],
            'Signal': 'Close Short Spread',
            'Long Asset': 'MSFT',
            'Short Asset': 'AAPL',
            'Exit Price Long': exit_price_MSFT,
            'Exit Price Short': exit_price_AAPL,
            'PnL': pnl_trade
        })
        
        #reset position
        position = 0
        long = 0
        short = 0

# Final Portfolio Analysis
returns['PnL'] = pd.Series(pnl).fillna(0.0)
returns['Cumulative_PnL'] = pd.Series(cumulative_pnl).fillna(0.0)

#summarize P&L
total_pnl = sum(pnl)
print(f"Total P&L: {total_pnl: .2f}")

#Plot the equity curve 
# Plot the equity curve
fig_equity = px.line(
    returns, 
    x='time', 
    y='Cumulative_PnL', 
    title='Equity Curve Over Time', 
    labels={'Cumulative_PnL': 'Cumulative P&L ($)', 'time': 'Date'}
)
fig_equity.show()

# Convert the log to a DataFrame
position_df = pd.DataFrame(position_log)
position_16 = position_df.tail(16)
print(position_16)
print('---')

#see cumulative trade
print(returns)
print(returns[['time', 'Spread_RSI']].tail(20))