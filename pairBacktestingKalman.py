import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pairAnalysis import calExpMovingAverages, linearRegressionSlope
from pykalman import KalmanFilter

plot_diff, plot_lb, plot_ub, spreads, normalized_spreads, res_stds = [], [], [], [], [], []
normalization_lookback = 100
transaction_costs = []


def regressionOutput(x, slope, intercept):
    return slope * x + intercept


def calSharpeRatio(percentage_returns):
    return np.mean(percentage_returns)/np.std(percentage_returns)


def calCompoundReturn(rel_returns):
    mul = 1
    for r in rel_returns:
        mul *= (1 + r)

    return mul - 1


def currentPos(x, y, reg_slope, lower_bound, upper_bound, mean):
    y_pred = regressionOutput(
        x, reg_slope, 0)
    print(y_pred, y, reg_slope)
    plot_lb.append(lower_bound)
    plot_ub.append(upper_bound)
    spreads.append(y-y_pred)

    normalized_spread = (y-y_pred)/np.std(spreads[-normalization_lookback:])
    normalized_spreads.append(normalized_spread)

    if spreads[-1] > upper_bound:
        return 2
    elif spreads[-1] < lower_bound:
        return -2
    elif spreads[-1] > mean:
        return 1
    else:
        return -1


def kalmanUpdate(x, y, theta, P, W, sigma_e, num_obs):

    theta_1 = theta
    P_1 = P + W
    print("Prior Theta and P", theta_1, P_1)

    y_1_tilde = y - np.dot(x, theta_1)
    print("Residual mean", y_1_tilde)

    # residual covariance
    V_1 = np.eye(num_obs)*sigma_e
    S_1 = np.dot(x, P_1)
    S_1 = np.dot(S_1, np.transpose(x))
    S_1 = S_1 + V_1
    # S_1 = np.dot(np.dot(x, P_1), np.transpose(x)) + V_1
    print("Residual covariance", S_1)

    # Kalman Gain
    K_1 = np.dot(np.dot(P_1, np.transpose(x)), np.linalg.inv(S_1))
    print("Kalman Gain", K_1)

    # Posterior
    theta_1 = theta_1 + np.dot(K_1, y_1_tilde)
    P_1 = P_1 - np.dot(np.dot(K_1, x), P_1)

    print("Posterior Theta and P", theta_1, P_1)
    return theta_1, P_1, S_1


def backtestKalman(initial_train_df1, initial_train_df2, test_df1, test_df2, ma_window_size, lower_bound, upper_bound, window_time, waiting_time, slope_diff_threshold, stop_loss, start_trading_threshold):
    # ma1, ma2 = calExpMovingAverages(
    #     initial_train_df1, initial_train_df2, ma_window_size)

    initial_theta = np.array([0.5]).reshape(1, 1)
    initial_P = np.array([1e-6]).reshape(1, 1)
    W = np.array([1e-6]).reshape(1, 1)
    sigma_e = 3.0

    theta = initial_theta
    P = initial_P

    reg_slope = linearRegressionSlope(
        initial_train_df1.iloc[:, 4], initial_train_df2.iloc[:, 4])
    
    # Trading States
    position = 0
    returns, percentage_returns = [], []
    is_open = False
    open_time = 0
    close_durations = []
    open_times = []
    cum_slope_diff = 0
    prev_slope = reg_slope
    mean = 0
    stop_trading = False
    stop_trading_days = 0
    c_stop = 0
    for i in range(1, len(test_df1)):
        print("Checking bar", i)
        price1 = test_df1.iloc[i, 1]
        price2 = test_df2.iloc[i, 1]
        print("Original prices", price1, price2)

        # Checking Kalman update condition 
        if i % window_time == 0 and i + window_time < len(test_df1):
            print("Updating Kalman Filter")
            theta, P, S = kalmanUpdate(np.array(test_df1.iloc[i-window_time:i, 4]).reshape(
                window_time, 1), np.array(test_df2.iloc[i-window_time:i, 4]).reshape(window_time, 1), theta, P, W, sigma_e, window_time)

            res_covariance = S[0][0]

            prev_slope = reg_slope
            reg_slope = theta[0][0]

        if stop_trading:
            stop_trading_days += 1
            if stop_trading_days == start_trading_threshold:
                stop_trading = False
                stop_trading_days = 0
            continue

        res_stds.append(res_covariance**(0.5))
        if is_open:
            return_till_now = position*open_slope * \
                (price1 - open1) + position*(open2 - price2)
            cum_slope_diff += (reg_slope - prev_slope)
            open_time += 1
            if np.abs(cum_slope_diff) > slope_diff_threshold:
                # if open_time == waiting_time:
                print("##############\nClose position", i)
                close1 = price1
                close2 = price2
                print("Close Prices", close1, close2)
                rel_return1 = position*open_slope*(close1 - open1)
                rel_return2 = position*(open2 - close2)
                rel_return = rel_return1 + rel_return2
                percentage_return = rel_return1 / \
                    (open_slope*open1) + rel_return2/(open2)
                print(rel_return1/(open_slope*open1), rel_return2/(open2))
                # if percentage_return > 0:
                percentage_returns.append(percentage_return)
                print(rel_return1, rel_return2)
                returns.append(rel_return)
                close_durations.append(open_time)
                position = 0
                is_open = False
                open_time = 0
                cum_slope_diff = 0
                transaction_costs.append(0.001*(open_slope*close1+close2))
                stop_trading = True
                c_stop += 1
                continue

            current_state = currentPos(
                price1, price2, open_slope, lower_bound, upper_bound, mean)
        else:
            current_state = currentPos(
                price1, price2, reg_slope, lower_bound, upper_bound, mean)
        print(current_state)

        # Checking for Opening and Closing conditions
        if current_state == 2 and position == 0:
            print("##############\nAbove line Open", i)
            # short ETH, long BTC
            open1 = price1
            open2 = price2
            print("Open Prices", open1, open2)
            open_times.append(i)
            position = 1
            is_open = True
            open_slope = reg_slope
            transaction_costs.append(0.001*(open_slope*open1+open2))
        elif current_state == -2 and position == 0:
            print("##############\nBelow line Open", i)
            # short BTC, long ETH
            open1 = price1
            open2 = price2
            print("Open Prices", open1, open2)
            open_times.append(i)
            position = -1
            is_open = True
            open_slope = reg_slope
            transaction_costs.append(0.001*(open_slope*open1+open2))
        elif (current_state == -1 or current_state == -2) and position == 1:
            print("##############\nClose position", i)
            # close all positions
            close1 = price1
            close2 = price2
            print("Close Prices", close1, close2)
            rel_return1 = open_slope*(close1 - open1)
            rel_return2 = (open2 - close2)
            rel_return = rel_return1 + rel_return2
            percentage_return = rel_return1 / \
                (open_slope*open1) + rel_return2/(open2)
            print(rel_return1/(open_slope*open1), rel_return2/(open2))
            percentage_returns.append(percentage_return)
            print(rel_return1, rel_return2)
            returns.append(rel_return)
            close_durations.append(open_time)
            position = 0
            is_open = False
            open_time = 0
            cum_slope_diff = 0
            transaction_costs.append(0.001*(open_slope*close1+close2))
        elif (current_state == 1 or current_state == 2) and position == -1:
            print("##############\nClose position", i)
            # close all positions
            close1 = price1
            close2 = price2
            print("Close Prices", close1, close2)
            rel_return1 = -open_slope*(close1 - open1)
            rel_return2 = -(open2 - close2)
            rel_return = rel_return1 + rel_return2
            percentage_return = rel_return1 / \
                (open_slope*open1) + rel_return2/(open2)
            print(rel_return1/(open_slope*open1), rel_return2/(open2))
            percentage_returns.append(percentage_return)
            print(rel_return1, rel_return2)
            returns.append(rel_return)
            close_durations.append(open_time)
            position = 0
            is_open = False
            open_time = 0
            cum_slope_diff = 0
            transaction_costs.append(0.001*(open_slope*close1+close2))

    return np.array(returns), open_times, close_durations, np.array(percentage_returns)


if __name__ == "__main__":
    initial_train_df1 = pd.concat([
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2021-09.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2021-10.csv", header=None)], ignore_index=True)

    initial_train_df2 = pd.concat([
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2021-09.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2021-10.csv", header=None)], ignore_index=True)

    test_df1 = pd.concat([
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2021-11.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2021-12.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2022-01.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2022-02.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2022-03.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2022-04.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2022-05.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2022-06.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2022-07.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2022-08.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2022-09.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2022-10.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2022-11.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2022-12.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2023-01.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2023-02.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2023-03.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2023-04.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2023-05.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2023-06.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/BTCUSDT-1m-2023-07.csv", header=None)
    ], ignore_index=True)

    test_df2 = pd.concat([
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2021-11.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2021-12.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2022-01.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2022-02.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2022-03.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2022-04.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2022-05.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2022-06.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2022-07.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2022-08.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2022-09.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2022-10.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2022-11.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2022-12.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2023-01.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2023-02.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2023-03.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2023-04.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2023-05.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2023-06.csv", header=None),
        pd.read_csv(
            "pair_analysis_data/minute/ETHUSDT-1m-2023-07.csv", header=None)
    ], ignore_index=True)

    print(len(initial_train_df1), len(test_df1))
    ma_window_size = 1
    lower_bound = -0.05
    upper_bound = 0.05
    window_time = 1
    waiting_time = 1000
    slope_diff_threshold = 0.003
    stop_loss = 200
    start_trading_threshold = 50


    returns, open_times, close_durations, percentage_returns = backtestKalman(
        initial_train_df1, initial_train_df2, test_df1, test_df2, ma_window_size, lower_bound, upper_bound, window_time, waiting_time, slope_diff_threshold, stop_loss, start_trading_threshold)
    for i in range(len(returns)):
        # if percentage_returns[i] < 0:
        print(returns[i], open_times[i], close_durations[i],
              percentage_returns[i] * 100)
    # print(list(zip(returns, open_times, close_durations, percentage_returns * 100)))
    print("Number of positions closed", len(returns))
    print("Total absolute return", sum(returns))
    print("Total transaction costs", sum(transaction_costs))
    print("Total Simple Return", sum(percentage_returns)*100)
    print("Total Compounded Return", calCompoundReturn(percentage_returns)*100)

    percentage_returns = percentage_returns * 100

    print("Sharpe Ratio", calSharpeRatio(percentage_returns))