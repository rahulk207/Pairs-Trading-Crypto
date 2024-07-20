import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression


def calExpMovingAverages(df1, df2, window_size):
    ma1 = list(df1.iloc[:, 4].ewm(span=window_size).mean())[window_size:]
    ma2 = list(df2.iloc[:, 4].ewm(span=window_size).mean())[window_size:]

    return ma1, ma2


def calMovingAverages(df1, df2, window_size):
    ma1 = list(df1.iloc[:, 4].rolling(window=window_size).mean()[window_size:])
    ma2 = list(df2.iloc[:, 4].rolling(window=window_size).mean()[window_size:])

    return ma1, ma2


def plotMovingAverages(ma1, ma2):
    plt.scatter(ma1, ma2)


def linear(x):
    return slope * x + intercept


def linearRegression(ma1, ma2):
    return stats.linregress(ma1, ma2)


def linearRegressionSlope(ma1, ma2):
    ma1 = [[x] for x in ma1]
    ma2 = [[x] for x in ma2]
    model = LinearRegression(fit_intercept=False)
    model.fit(ma1, ma2)

    return model.coef_[0][0]


if __name__ == '__main__':
    # df1 = pd.concat([pd.read_csv("pair_analysis/BTCUSDT-1d-2023-01.csv", header=None),
    #                  pd.read_csv(
    #                      "pair_analysis/BTCUSDT-1d-2023-02.csv", header=None),
    #                  pd.read_csv(
    #                      "pair_analysis/BTCUSDT-1d-2023-03.csv", header=None),
    #                  pd.read_csv(
    #                      "pair_analysis/BTCUSDT-1d-2023-04.csv", header=None),
    #                  pd.read_csv("pair_analysis/BTCUSDT-1d-2023-05.csv", header=None)], ignore_index=True)

    # df2 = pd.concat([pd.read_csv("pair_analysis/ETHUSDT-1d-2023-01.csv", header=None),
    #                  pd.read_csv(
    #                      "pair_analysis/ETHUSDT-1d-2023-02.csv", header=None),
    #                  pd.read_csv(
    #                      "pair_analysis/ETHUSDT-1d-2023-03.csv", header=None),
    #                  pd.read_csv(
    #                      "pair_analysis/ETHUSDT-1d-2023-04.csv", header=None),
    #                  pd.read_csv("pair_analysis/ETHUSDT-1d-2023-05.csv", header=None)], ignore_index=True)

    df1 = pd.concat([
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-01.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-02.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-03.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-04.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-05.csv", header=None),
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
            "pair_analysis/BTCUSDT-1d-2021-12.csv", header=None),], ignore_index=True)

    df2 = pd.concat([
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-01.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-02.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-03.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-04.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-05.csv", header=None),
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
            "pair_analysis/ETHUSDT-1d-2021-12.csv", header=None)], ignore_index=True)

    ma1, ma2 = calMovingAverages(df1, df2, 10)
    # plotMovingAverages(ma1, ma2)

    regressor = linearRegression(ma1, ma2)
    slope, intercept, r, p = regressor.slope, regressor.intercept, regressor.rvalue, regressor.pvalue
    slope_std_err, intercept_std_err = regressor.stderr, regressor.intercept_stderr

    print(slope, intercept, slope_std_err, intercept_std_err)

    reg_main = list(map(linear, ma1))
    plt.plot(ma1, reg_main)

    slope += slope_std_err
    intercept += intercept_std_err
    reg_std_dev_plus = list(map(linear, ma1))
    plt.plot(ma1, reg_std_dev_plus, linestyle="dashed")

    slope -= 2*slope_std_err
    intercept -= 2*intercept_std_err
    reg_std_dev_minus = list(map(linear, ma1))
    plt.plot(ma1, reg_std_dev_minus, linestyle="dashed")

    df3 = pd.concat([
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-01.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-02.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-03.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-04.csv", header=None),
        pd.read_csv(
            "pair_analysis/BTCUSDT-1d-2021-05.csv", header=None),
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
            "pair_analysis/BTCUSDT-1d-2021-12.csv", header=None),], ignore_index=True)

    df4 = pd.concat([
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-01.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-02.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-03.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-04.csv", header=None),
        pd.read_csv(
            "pair_analysis/ETHUSDT-1d-2021-05.csv", header=None),
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
            "pair_analysis/ETHUSDT-1d-2021-12.csv", header=None)], ignore_index=True)

    ma3, ma4 = calMovingAverages(df3, df4, 10)
    labels = list(range(len(ma3)))

    plt.scatter(ma3, ma4)

    for i, txt in enumerate(labels):
        plt.annotate(txt, (ma3[i], ma4[i]))

    plt.show()
