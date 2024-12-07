# Statistical Arbitrage Project

This project implements a Statistical Arbitrage trading strategy by analyzing the co-integration of two correlated assets: Apple Inc. (AAPL) and Microsoft Corp. (MSFT). The strategy is based on spread trading, where buy/sell signals are generated using Bollinger Bands and Relative Strength Index (RSI) to determine entry and exit points.
Additionally, a backtesting framework is included to evaluate the profitability of the trading signals by simulating long and short positions.

## Features
1. Data Retrieval and Preprocessing
- Data Source: Stock price data is sourced from Yahoo Finance using the yfinance library.
- Customizable Parameters: Specify desired tickers, date ranges, and other strategy parameters.
- Daily Returns Calculation: Converts stock price data into daily percentage returns for both securities.

2. Statistical Tests
- Ordinary Least Squares (OLS) Regression: Estimates the relationship between the two securities and computes the beta coefficient for spread calculation.
- Augmented Dickey-Fuller (ADF) Test: Tests the stationarity of the residuals to confirm co-integration. Results determine whether the pair is suitable for pairs trading.

3. Bollinger Bands and Relative Strength Index (RSI) for Spread Analysis
- Dynamic Band Calculation: This method computes the spread's upper, lower, and mean bands using a rolling window. Configurable parameters include window size and standard deviation multiplier.
- Integration with Trading Signals: Signals are generated when the spread crosses these bands, indicating potential entry and exit points.
- Custom RSI Calculation: Measures the momentum of the spread to provide additional confirmation for trade signals.

4. Trade Signal Generation
- Buy Signal: Triggered when the spread crosses below the lower Bollinger Band and the RSI is below the oversold threshold.
- Sell Signal: Triggered when the spread crosses above the upper Bollinger Band and the RSI is above the overbought threshold.
- Signal Visualization: Markers on the spread chart indicate buy and sell signals for easy interpretation.

5. Backtesting Framework
- PnL Calculation: Simulates long and short trades based on signals, calculates profit and loss (PnL) for each trade, and tracks cumulative PnL over time.
- Transaction Costs: Incorporates customizable transaction cost rates to provide realistic performance metrics.
- Position Management: Tracks open positions and ensures trades are executed when the spread crosses predefined thresholds.
- Position Tracking: Logs the time, trade type (long or short), entry and exit prices, and resulting PnL for every trade.

6. Interactive Data Visualization
- Bollinger Band Visualization: This tool displays the spread with upper, lower, and mean bands along with buy and sell signals.
- Interactive RSI Visualization: Plots RSI values over time with overbought (67) and oversold (33) thresholds for better decision-making.
- PnL Visualization: Future enhancements could include cumulative PnL charts to track strategy performance over time.

## Blog
My [Medium](https://medium.com/@napoles.jair/statistical-arbitrage-3e8134d1efb3) post provides a detailed explanation of my project's implementation of pairs trading.
