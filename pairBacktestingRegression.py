import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pairAnalysis import calExpMovingAverages, linearRegression


def regressionOutput(x, slope, intercept):
    return slope * x + intercept


def calSharpeRatio(percentage_returns):
    return np.mean(percentage_returns)/np.std(percentage_returns)


def currentPos(x, y, reg_slope, reg_intercept, reg_slope_err, reg_intercept_err):
    upper_bound = regressionOutput(
        x, reg_slope + reg_slope_err, reg_intercept + reg_intercept_err)
    lower_bound = regressionOutput(
        x, reg_slope - reg_slope_err, reg_intercept - reg_intercept_err)

    if y > upper_bound:
        return 1
    elif y < lower_bound:
        return -1
    else:
        return 0


def backtest(df1, df2, window_size, reg_slope, reg_intercept, reg_slope_err, reg_intercept_err):
    ma1, ma2 = calExpMovingAverages(df1, df2, window_size)
    df1 = df1.iloc[window_size:, :]
    df2 = df2.iloc[window_size:, :]
    prev_state = 0
    returns = []
    for i in range(len(df1)):
        price1 = df1.iloc[i, 4]
        price2 = df2.iloc[i, 4]
        current_state = currentPos(
            ma1[i], ma2[i], reg_slope, reg_intercept, reg_slope_err, reg_intercept_err)
        if current_state == 1 and prev_state == 0:
            print("Above line Open", i)
            # short ETH, long BTC
            open1 = price1
            open2 = price2
        elif current_state == -1 and prev_state == 0:
            print("Below line Open", i)
            # short BTC, long ETH
            open1 = price1
            open2 = price2
        elif current_state == 0 and prev_state != 0:
            print("Close position", i)
            # close all positions
            close1 = price1
            close2 = price2
            rel_return1 = prev_state * (close1 - open1)/open1
            rel_return2 = prev_state * (open2 - close2)/open2
            rel_return = rel_return1 + rel_return2
            print(rel_return1, rel_return2)
            returns.append(rel_return)

        prev_state = current_state

    return returns


if __name__ == "__main__":
    train_df1 = pd.concat([
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-01.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-02.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-03.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-04.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-05.csv", header=None)], ignore_index=True)
    train_df2 = pd.concat([
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-01.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-02.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-03.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-04.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-05.csv", header=None)], ignore_index=True)

    test_df1 = pd.concat([
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-06.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-07.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-08.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-09.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-10.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-11.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-12.csv", header=None)], ignore_index=True)

    test_df2 = pd.concat([
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-06.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-07.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-08.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-09.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-10.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-11.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-12.csv", header=None)], ignore_index=True)

    ma_window_size = 10
    ma1, ma2 = calExpMovingAverages(train_df1, train_df2, ma_window_size)
    regressor = linearRegression(ma1, ma2)
    reg_slope, reg_intercept, reg_slope_err, reg_intercept_err = regressor.slope, regressor.intercept, regressor.stderr, regressor.intercept_stderr

    returns = backtest(
        test_df1, test_df2, ma_window_size, reg_slope, reg_intercept, reg_slope_err, reg_intercept_err)
    print(returns)
    print("Number of positions closed", len(returns))
    print("Total absolute return", sum(returns))

    percentage_returns = returns * 100

    print("Sharpe Ratio", calSharpeRatio(percentage_returns))

    # Write continual training code. Two hyperparameters - (1) how many months to consider for training - should be small (2-3 months),
    # (2) Waiting time, after which we need to retrain the regression line.
    # Let's calculate sharpe ratio and returns for all 3 years of data with this strategy.

    # Try the same algorithm with exponential-MA instead of normal MA.
