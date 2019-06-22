#https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py
import pandas as pd
import numpy as np

def relative_strength_index(df, period):
    """Calculate Relative Strength Index(RSI) for given data.
    https://pocketsense.com/calculate-stocks-relative-strength-5178.html
    :param df: pandas.DataFrame
    :param period: period of days
    :return: pandas.DataFrame
    """

    diff = df['Close'].diff()
    diff[0] = 0;  # replace NaN
    RS = []
    for i in range(0, len(diff)):
        total_gain = 0
        total_loss = 1e-10  # prevent division by zero
        for j in range(1, period+1):
            if i-j < 0:
                value = 0
            else:
                value = diff[i-j]
            if value < 0:
                total_loss += value
            else:
                total_gain += value
        avgGain = total_gain / period
        avgLoss = total_loss / period
        RS.append(abs(avgGain / avgLoss))
    RSI = [(100 - (100 / (1 + rs))) for rs in RS]
    return pd.Series(RSI, name='RSI')


def stochastic_oscillator(df, period):
    """Calculate stochastic oscillator %K for given data.
    """
    H_period = df['High'].rolling(window=period).max()
    L_period = df['Low'].rolling(window=period).min()
    SOk = (df['Close'] - L_period) / (H_period - L_period) * 100
    SOk.name = '%K'
    return SOk

def williams(df, period):
    """Calculate Williams %R for given data.
    """
    H_period = df['High'].rolling(window=period).max()
    L_period = df['Low'].rolling(window=period).min()
    williams = (H_period - df['Close']) / (H_period - L_period) * -100
    williams.name = '%R'
    return williams

def exponential_moving_average(series, n):
    k = 2 / (1 + n)
    ema = (series * k) + (series.shift(1) * (1 - k))
    return ema

def moving_average_convergence_divergence(df):
    macd = exponential_moving_average(df['Close'], n=12) - exponential_moving_average(df['Close'], n=26)
    macd.name = 'MACD'
    signal_line = exponential_moving_average(macd, n=9)
    signal_line.name = 'SignalLine'
    return (macd, signal_line)

def price_rate_of_change(df, n):
    proc = df['Close'].diff(periods=n) / df['Close'].shift(periods=n)
    proc.name = 'PROC'
    return proc

def on_balance_volume(df):
    obv = [0]
    for t in range(1, len(df)):
        C_t = df['Close'][t]
        C_before_t = df['Close'][t-1]
        Vol_t = df['Volume'][t]
        if C_t > C_before_t:
            obv.append(obv[t-1] + Vol_t)
        elif C_t < C_before_t:
            obv.append(obv[t-1] - Vol_t)
        else:
            obv.append(obv[t-1])
    return pd.Series(obv, name='OBV')