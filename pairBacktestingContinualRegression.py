import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pairAnalysis import calExpMovingAverages, linearRegressionSlope

plot_y, plot_diff, plot_y_pred, plot_lb, plot_ub = [], [], [], [], []


def regressionOutput(x, slope, intercept):
    return slope * x + intercept


def calSharpeRatio(percentage_returns):
    return np.mean(percentage_returns)/np.std(percentage_returns)


def currentPos(x, y, reg_slope, lower_bound, upper_bound):
    print(len(plot_y))
    y_pred = regressionOutput(
        x, reg_slope, 0)

    plot_y.append(y)
    plot_y_pred.append(y_pred)
    plot_lb.append(lower_bound)
    plot_ub.append(upper_bound)

    if y - y_pred > upper_bound:
        return 2
    elif y - y_pred < lower_bound:
        return -2
    elif y - y_pred > 0:
        return 1
    else:
        return -1


def backtest(df1, df2, window_size, reg_slope, lower_bound, upper_bound, open1, open2):
    ma1, ma2 = calExpMovingAverages(df1, df2, window_size)
    df1 = df1.iloc[window_size:, :]
    df2 = df2.iloc[window_size:, :]
    # ma1 = list(df1.iloc[:, 4])
    # ma2 = list(df2.iloc[:, 4])
    position = 0
    returns = []
    for i in range(len(df1)):
        price1 = df1.iloc[i, 4]
        price2 = df2.iloc[i, 4]
        current_state = currentPos(
            ma1[i], ma2[i], reg_slope, lower_bound, upper_bound)
        if current_state == 2 and position == 0:
            print("Above line Open", i)
            # short ETH, long BTC
            open1 = price1
            open2 = price2
            print(ma1[i], ma2[i])
            print(open1, open2)
            position = 1
        elif current_state == -2 and position == 0:
            print("Below line Open", i)
            # short BTC, long ETH
            open1 = price1
            open2 = price2
            print(ma1[i], ma2[i])
            print(open1, open2)
            position = -1
        elif (current_state == -1 or current_state == -2) and position == 1:
            print("Close position", i)
            # close all positions
            close1 = price1
            close2 = price2
            print(ma1[i], ma2[i])
            print(close1, close2)
            rel_return1 = reg_slope*(close1 - open1)
            rel_return2 = (open2 - close2)
            rel_return = rel_return1 + rel_return2
            print(rel_return1, rel_return2)
            returns.append(rel_return)
            position = 0
        elif (current_state == 1 or current_state == 2) and position == -1:
            print("Close position", i)
            # close all positions
            close1 = price1
            close2 = price2
            print(ma1[i], ma2[i])
            print(close1, close2)
            rel_return1 = -reg_slope*(close1 - open1)
            rel_return2 = -(open2 - close2)
            rel_return = rel_return1 + rel_return2
            print(rel_return1, rel_return2)
            returns.append(rel_return)
            position = 0

            # if rel_return < 0:
            #     reg_main = regressionOutput(
            #         np.array(ma1), reg_slope, reg_intercept)
            #     plt.plot(ma1, reg_main)

            #     reg_std_dev_plus = regressionOutput(
            #         np.array(ma1), reg_slope + reg_slope_err, reg_intercept + reg_intercept_err)
            #     plt.plot(ma1, reg_std_dev_plus, linestyle="dashed")

            #     reg_std_dev_minus = regressionOutput(
            #         np.array(ma1), reg_slope - reg_slope_err, reg_intercept - reg_intercept_err)
            #     plt.plot(ma1, reg_std_dev_minus, linestyle="dashed")

            #     labels = list(range(len(ma1)))
            #     plt.scatter(ma1, ma2)

            #     for i, txt in enumerate(labels):
            #         plt.annotate(txt, (ma1[i], ma2[i]))

            #     plt.show()

        # prev_state = current_state

    return returns, position, open1, open2


def continualTrainBacktest(initial_train_df1, initial_train_df2, test_df1, test_df2, ma_window_size, waiting_time, continual_time, close_waiting_time, lower_bound, upper_bound):
    ma1, ma2 = calExpMovingAverages(
        initial_train_df1, initial_train_df2, ma_window_size)
    # ma1 = list(initial_train_df1.iloc[:, 4])
    # ma2 = list(initial_train_df2.iloc[:, 4])
    slope = linearRegressionSlope(ma1, ma2)
    # slope, intercept, slope_std_err, intercept_std_err = regressor.slope, regressor.intercept, regressor.stderr, regressor.intercept_stderr

    returns = []
    open1 = 0
    open2 = 0
    pos = len(initial_train_df1)
    while pos < len(test_df1):
        print("Regression Slope", slope)
        returns_current, position, open1, open2 = backtest(test_df1.iloc[pos-ma_window_size:pos+waiting_time, :], test_df2.iloc[pos-ma_window_size:pos+waiting_time, :],
                                                           ma_window_size, slope, lower_bound, upper_bound, open1, open2)

        returns.extend(returns_current)

        pos += waiting_time

        if pos >= len(test_df1) and position != 0:
            # print("Close position", pos)
            # # close all positions
            # close1 = test_df1.iloc[-1, 4]
            # close2 = test_df2.iloc[-1, 4]
            # rel_return1 = prev_state * (close1 - open1)/open1
            # rel_return2 = prev_state * (open2 - close2)/open2
            # rel_return = rel_return1 + rel_return2
            # print(rel_return1, rel_return2)
            # returns.append(rel_return)
            break

        c = 0
        # If the last open position exceeds len(df), we'll just keep it open for backtesting purposes
        while position != 0 and c < close_waiting_time and pos < len(test_df1):
            print(pos)
            price1 = test_df1.iloc[pos, 4]
            price2 = test_df2.iloc[pos, 4]
            ma1, ma2 = calExpMovingAverages(
                test_df1.iloc[pos-ma_window_size:pos+1, :], test_df2.iloc[pos-ma_window_size:pos+1], ma_window_size)
            current_state = currentPos(
                ma1[0], ma2[0], slope, lower_bound, upper_bound)
            c += 1

            if ((current_state == -1 or current_state == -2) and position == 1) or c == close_waiting_time:
                print("Close position", pos)
                # close all positions
                close1 = price1
                close2 = price2
                print(ma1[0], ma2[0])
                print(close1, close2)
                rel_return1 = slope*(close1 - open1)
                rel_return2 = (open2 - close2)
                rel_return = rel_return1 + rel_return2
                print(rel_return1, rel_return2)
                returns.append(rel_return)
                position = 0

            elif ((current_state == 1 or current_state == 2) and position == -1) or c == close_waiting_time:
                print("Close position", pos)
                # close all positions
                close1 = price1
                close2 = price2
                print(ma1[0], ma2[0])
                print(close1, close2)
                rel_return1 = -slope*(close1 - open1)
                rel_return2 = -(open2 - close2)
                rel_return = rel_return1 + rel_return2
                print(rel_return1, rel_return2)
                returns.append(rel_return)
                position = 0

            pos += 1

        if pos < len(test_df1):
            print("Trained again")
            ma1, ma2 = calExpMovingAverages(
                test_df1.iloc[pos-continual_time:pos, :], test_df2.iloc[pos-continual_time:pos, :], ma_window_size)
            # ma1 = list(test_df1.iloc[pos-continual_time:pos, :])
            # ma2 = list(test_df2.iloc[pos-continual_time:pos, :])
            slope = linearRegressionSlope(ma1, ma2)
            # slope, intercept, slope_std_err, intercept_std_err = regressor.slope, regressor.intercept, regressor.stderr, regressor.intercept_stderr
            # print("Slope", slope, "Intercept", intercept)

    return returns


if __name__ == "__main__":
    initial_train_df1 = pd.concat([
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-06.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-07.csv", header=None)], ignore_index=True)
    # pd.read_csv("pair_analysis/BTCUSDT-1d-2020-03.csv", header=None),
    # pd.read_csv("pair_analysis/BTCUSDT-1d-2020-04.csv", header=None),
    # pd.read_csv("pair_analysis/BTCUSDT-1d-2020-05.csv", header=None),
    # pd.read_csv("pair_analysis/BTCUSDT-1d-2020-06.csv", header=None),
    # pd.read_csv("pair_analysis/BTCUSDT-1d-2020-07.csv", header=None),
    # pd.read_csv("pair_analysis/BTCUSDT-1d-2020-08.csv", header=None),
    # pd.read_csv("pair_analysis/BTCUSDT-1d-2020-09.csv", header=None),
    # pd.read_csv("pair_analysis/BTCUSDT-1d-2020-10.csv", header=None),
    # pd.read_csv("pair_analysis/BTCUSDT-1d-2020-11.csv", header=None),
    # pd.read_csv("pair_analysis/BTCUSDT-1d-2020-12.csv", header=None)], ignore_index=True)
    initial_train_df2 = pd.concat([
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-06.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-07.csv", header=None)], ignore_index=True)
    # pd.read_csv("pair_analysis/ETHUSDT-1d-2020-03.csv", header=None),
    # pd.read_csv("pair_analysis/ETHUSDT-1d-2020-04.csv", header=None),
    # pd.read_csv("pair_analysis/ETHUSDT-1d-2020-05.csv", header=None),
    # pd.read_csv("pair_analysis/ETHUSDT-1d-2020-06.csv", header=None),
    # pd.read_csv("pair_analysis/ETHUSDT-1d-2020-07.csv", header=None),
    # pd.read_csv("pair_analysis/ETHUSDT-1d-2020-08.csv", header=None),
    # pd.read_csv("pair_analysis/ETHUSDT-1d-2020-09.csv", header=None),
    # pd.read_csv("pair_analysis/ETHUSDT-1d-2020-10.csv", header=None),
    # pd.read_csv("pair_analysis/ETHUSDT-1d-2020-11.csv", header=None),
    # pd.read_csv("pair_analysis/ETHUSDT-1d-2020-12.csv", header=None)], ignore_index=True)

    test_df1 = pd.concat([
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-06.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-07.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-08.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-09.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-10.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-11.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-12.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-01.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-02.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-03.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-04.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2022-05.csv", header=None),
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
            "pair_analysis/BTCUSDT-1d-2022-12.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2023-01.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2023-02.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2023-03.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2023-04.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2023-05.csv", header=None)], ignore_index=True)

    test_df2 = pd.concat([
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-06.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-07.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-08.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-09.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-10.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-11.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-12.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-01.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-02.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-03.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-04.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2022-05.csv", header=None),
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
            "pair_analysis/ETHUSDT-1d-2022-12.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2023-01.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2023-02.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2023-03.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2023-04.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2023-05.csv", header=None)], ignore_index=True)

    print(len(initial_train_df1), len(test_df1))
    ma_window_size = 1
    waiting_time = 1000  # in days
    continual_time = 10  # in days
    close_waiting_time = 60
    lower_bound = -50
    upper_bound = 50

    returns = continualTrainBacktest(
        initial_train_df1, initial_train_df2, test_df1, test_df2, ma_window_size, waiting_time, continual_time, close_waiting_time, lower_bound, upper_bound)
    print(returns)
    print("Number of positions closed", len(returns))
    print("Total absolute return", sum(returns))

    percentage_returns = returns * 100

    print("Sharpe Ratio", calSharpeRatio(percentage_returns))

    # Write continual training code. Two hyperparameters - (1) how many months to consider for training - should be small (2-3 months),
    # (2) Waiting time, after which we need to retrain the regression line.
    # Let's calculate sharpe ratio and returns for all 3 years of data with this strategy.

    # Try the same algorithm with exponential-MA instead of normal MA.

    plt.xticks(np.arange(0, len(plot_y), 30))

    plt.plot([y-y_pred for y, y_pred in zip(plot_y, plot_y_pred)])
    plt.plot([y_ub for y_ub in plot_ub])
    plt.plot([y_lb for y_lb in plot_lb])
    # plt.plot(plot_lb, linestyle="dotted")
    # plt.plot(plot_ub, linestyle="dotted")

    plt.show()
