import numpy as np
import random

np.random.seed(1)
random.seed(1)

import technical_indicators as ta

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, precision_score, confusion_matrix, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

# read data file
aapl = pd.read_csv('AAPL.csv')
del(aapl['Date'])
del(aapl['Adj Close'])

def main(days):
    # exponential smoothing
    smooth_data = aapl.ewm(alpha=0.5, adjust=False).mean()

    # feature extraction
    rsi = ta.relative_strength_index(smooth_data, period=14)
    stochastic_oscillator = ta.stochastic_oscillator(smooth_data, period=14)
    williams = ta.williams(smooth_data, period=14)
    (macd, signal_line) = ta.moving_average_convergence_divergence(smooth_data)
    proc = ta.price_rate_of_change(smooth_data, n=days)
    obv = ta.on_balance_volume(smooth_data)

    del (smooth_data['Open'])
    del (smooth_data['High'])
    del (smooth_data['Low'])
    del (smooth_data['Volume'])

    # create label column
    label = (smooth_data.shift(-days)['Close'] >= smooth_data['Close'])
    label = label.iloc[:-days]
    label = label.astype(int)
    label.name = 'Label'

    # final dataset
    dataset = smooth_data.join([rsi, stochastic_oscillator, williams, macd, signal_line, proc, obv, label]).dropna()
    del (dataset['Close'])

    # process dataset
    y = dataset['Label']
    features = [x for x in dataset.columns if x != 'Label']# lay ten nhung cot khong phai Label
    X = dataset[features]

    ## data leaked dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2 * len(X) // 3)#chia X lam X,Y lamf 3 phan,
    # 2 phan train, 1 phan test

    # train
    rf = RandomForestClassifier(n_jobs=-1, n_estimators=65, random_state=42)
    rf.fit(X_train, y_train.values.ravel());

    # test
    pred = rf.predict(X_test)
    precision = precision_score(y_pred=pred, y_true=y_test)
    recall = recall_score(y_pred=pred, y_true=y_test)
    f1 = f1_score(y_pred=pred, y_true=y_test)
    accuracy = accuracy_score(y_pred=pred, y_true=y_test)
    print('precision: {0:1.2f}, recall: {1:1.2f}, f1: {2:1.2f}, accuracy: {3:1.2f}'.format(precision, recall, f1,
                                                                                           accuracy))
for num_day in [30, 60, 90]:
    print('Trading Period =', num_day, 'days')
    main(days=num_day)
    print('------------------------------')