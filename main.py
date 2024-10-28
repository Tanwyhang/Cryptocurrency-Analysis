import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Retrieve Crypto Market Data
def get_crypto_data(ticker, start_date, end_date, interval='1h'):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            raise ValueError("No data returned from Yahoo Finance.")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of error

# Define Multiple Trading Strategies
def moving_average_strategy(data, short_window, long_window):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    
    signals['signal'] = 0.0
    signals.loc[signals['short_mavg'] > signals['long_mavg'], 'signal'] = 1.0
    signals.loc[signals['short_mavg'] <= signals['long_mavg'], 'signal'] = 0.0
    
    signals['positions'] = signals['signal'].diff().fillna(0)
    return signals

def momentum_strategy(data, window):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['momentum'] = data['Close'] - data['Close'].shift(window)
    signals['signal'] = np.where(signals['momentum'] > 0, 1.0, 0.0)
    signals['positions'] = signals['signal'].diff().fillna(0)
    return signals

def mean_reversion_strategy(data, window):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['mean'] = data['Close'].rolling(window=window).mean()
    signals['std'] = data['Close'].rolling(window=window).std()
    signals['z_score'] = (signals['price'] - signals['mean']) / signals['std']
    signals['signal'] = np.where(signals['z_score'] < -1, 1.0, np.where(signals['z_score'] > 1, -1.0, 0.0))
    signals['positions'] = signals['signal'].diff().fillna(0)
    return signals

# Backtesting Function
def backtest_strategy(data, signals):
    initial_capital = 10000.0
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['crypto'] = signals['signal'] * 1000
    
    # Portfolio value calculation
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['holdings'] = (positions['crypto'] * data['Close']).fillna(0.0)
    portfolio['cash'] = initial_capital - (positions['crypto'].diff().fillna(0) * data['Close']).cumsum().fillna(0.0)
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    
    return portfolio

# Optimize Strategy Parameters
def optimize_parameters(data):
    def objective(params):
        short_window, long_window, momentum_window, mean_window = params
        signals_ma = moving_average_strategy(data, int(short_window), int(long_window))
        signals_mo = momentum_strategy(data, int(momentum_window))
        signals_mr = mean_reversion_strategy(data, int(mean_window))
        
        portfolio_ma = backtest_strategy(data, signals_ma)
        portfolio_mo = backtest_strategy(data, signals_mo)
        portfolio_mr = backtest_strategy(data, signals_mr)
        
        combined_portfolio = (portfolio_ma['total'] + portfolio_mo['total'] + portfolio_mr['total']) / 3
        return -combined_portfolio.iloc[-1]  # Negative for minimization
    
    # Adjust parameter bounds for midterm trading
    result = minimize(objective, [30, 90, 20, 30], bounds=[(20, 50), (50, 200), (10, 30), (20, 60)])
    return result.x

# Plot Results for All Strategies
def plot_results(data, signals_ma, signals_mo, signals_mr, portfolio_ma, portfolio_mo, portfolio_mr):
    plt.figure(figsize=(16, 12))
    
    # Plot price and signals for Moving Average Strategy
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['Close'], label='Close Price', color='black', alpha=0.5)
    plt.plot(data.index, signals_ma['short_mavg'], label='Short Moving Average', color='blue')
    plt.plot(data.index, signals_ma['long_mavg'], label='Long Moving Average', color='red')
    plt.plot(signals_ma.loc[signals_ma.positions == 1.0].index, 
             signals_ma.short_mavg[signals_ma.positions == 1.0], 
             '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(signals_ma.loc[signals_ma.positions == -1.0].index, 
             signals_ma.short_mavg[signals_ma.positions == -1.0], 
             'v', markersize=10, color='r', lw=0, label='Sell Signal')
    plt.title('Moving Average Strategy')
    plt.legend()
    
    # Plot price and signals for Momentum Strategy
    plt.subplot(3, 1, 2)
    plt.plot(data.index, data['Close'], label='Close Price', color='black', alpha=0.5)
    plt.plot(signals_mo.loc[signals_mo.positions == 1.0].index, 
             signals_mo['price'][signals_mo.positions == 1.0], 
             '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(signals_mo.loc[signals_mo.positions == -1.0].index, 
             signals_mo['price'][signals_mo.positions == -1.0], 
             'v', markersize=10, color='r', lw=0, label='Sell Signal')
    plt.title('Momentum Strategy')
    plt.legend()
    
    # Plot price and signals for Mean Reversion Strategy
    plt.subplot(3, 1, 3)
    plt.plot(data.index, data['Close'], label='Close Price', color='black', alpha=0.5)
    plt.plot(signals_mr.loc[signals_mr.positions == 1.0].index, 
             signals_mr['price'][signals_mr.positions == 1.0], 
             '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(signals_mr.loc[signals_mr.positions == -1.0].index, 
             signals_mr['price'][signals_mr.positions == -1.0], 
             'v', markersize=10, color='r', lw=0, label='Sell Signal')
    plt.title('Mean Reversion Strategy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define parameters
    ticker = 'BTC-USD'
    start_date = '2023-01-01'  # Ensure the date range is within the last 730 days
    end_date = '2024-01-01'
    
    # Retrieve data
    data = get_crypto_data(ticker, start_date, end_date, interval='1h')
    
    if data.empty:
        print(f"No data available for {ticker} between {start_date} and {end_date}.")
    else:
        # Optimize strategy parameters
        optimal_params = optimize_parameters(data)
        print(f'Optimal Parameters: Short Window = {optimal_params[0]}, Long Window = {optimal_params[1]}, Momentum Window = {optimal_params[2]}, Mean Window = {optimal_params[3]}')
        
        # Generate signals with optimized parameters
        signals_ma = moving_average_strategy(data, int(optimal_params[0]), int(optimal_params[1]))
        signals_mo = momentum_strategy(data, int(optimal_params[2]))
        signals_mr = mean_reversion_strategy(data, int(optimal_params[3]))
        
        # Backtest strategies
        portfolio_ma = backtest_strategy(data, signals_ma)
        portfolio_mo = backtest_strategy(data, signals_mo)
        portfolio_mr = backtest_strategy(data, signals_mr)
        
        # Combine portfolios
        combined_portfolio = (portfolio_ma['total'] + portfolio_mo['total'] + portfolio_mr['total']) / 3
        
        # Plot results
        plot_results(data, signals_ma, signals_mo, signals_mr, portfolio_ma, portfolio_mo, portfolio_mr)
