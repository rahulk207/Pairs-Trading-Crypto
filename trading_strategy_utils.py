import numpy as np
import json


def regressionOutput(x, slope, intercept):
    return slope * x + intercept


def calSharpeRatio(percentage_returns):
    return np.mean(percentage_returns)/np.std(percentage_returns)


def calCompoundReturn(rel_returns):
    mul = 1
    for r in rel_returns:
        mul *= (1 + r)

    return mul - 1


def currentPos(x, y, reg_slope, lower_bound, upper_bound):
    y_pred = regressionOutput(
        x, reg_slope, 0)
    print(y_pred, y, reg_slope)
    spread = y-y_pred

    if spread > upper_bound:
        return 2
    elif spread < lower_bound:
        return -2
    elif spread > 0:
        return 1
    else:
        return -1


def kalmanUpdate(x, y, theta, P, W, sigma_e, num_obs):

    theta_1 = theta
    P_1 = P + W
    # print("Prior Theta and P", theta_1, P_1)

    y_1_tilde = y - np.dot(x, theta_1)
    # print("Residual mean", y_1_tilde)

    # residual covariance
    V_1 = np.eye(num_obs)*sigma_e
    S_1 = np.dot(x, P_1)
    S_1 = np.dot(S_1, np.transpose(x))
    S_1 = S_1 + V_1
    # S_1 = np.dot(np.dot(x, P_1), np.transpose(x)) + V_1
    # print("Residual covariance", S_1)

    # Kalman Gain
    K_1 = np.dot(np.dot(P_1, np.transpose(x)), np.linalg.inv(S_1))
    # print("Kalman Gain", K_1)

    # Posterior
    theta_1 = theta_1 + np.dot(K_1, y_1_tilde)
    P_1 = P_1 - np.dot(np.dot(K_1, x), P_1)

    # print("Posterior Theta and P", theta_1, P_1)
    return theta_1, P_1


# def fetchDailyBarAndTrade():
#     global data_btc
#     global data_eth
#     # Function to handle incoming messages from the WebSocket

#     def on_message_btc(ws, message):
#         data = json.loads(message)
#         timestamp = data['E']
#         current_time = datetime.datetime.now()
#         today_2pm = current_time.replace(
#             hour=14, minute=0, second=0, microsecond=0)
#         if not hasattr(on_message_btc, "has_crossed_2pm"):
#             on_message_btc.has_crossed_2pm = False
#         if datetime.datetime.fromtimestamp(timestamp / 1000.0) >= today_2pm and not on_message_btc.has_crossed_2pm:
#             on_message_btc.has_crossed_2pm = True
#             data_btc = data
#             logger.info(data_btc)
#             checkTradingLogic()
#         elif datetime.datetime.fromtimestamp(timestamp / 1000.0) < today_2pm:
#             on_message_btc.has_crossed_2pm = False

#     def on_message_eth(ws, message):
#         data = json.loads(message)
#         timestamp = data['E']
#         current_time = datetime.datetime.now()
#         today_2pm = current_time.replace(
#             hour=14, minute=0, second=0, microsecond=0)
#         if not hasattr(on_message_eth, "has_crossed_2pm"):
#             on_message_eth.has_crossed_2pm = False
#         if datetime.datetime.fromtimestamp(timestamp / 1000.0) >= today_2pm and not on_message_eth.has_crossed_2pm:
#             on_message_eth.has_crossed_2pm = True
#             data_eth = data
#             logger.info(data_eth)
#             checkTradingLogic()
#         elif datetime.datetime.fromtimestamp(timestamp / 1000.0) < today_2pm:
#             on_message_eth.has_crossed_2pm = False

#     # Function to handle errors from the WebSocket
#     def on_error(ws, error):
#         logger.error(error)

#     # Function to handle the WebSocket closing
#     def on_close(ws):
#         logger.info("WebSocket closed")
#         connect_to_websocket()  # reconnect to websocket when it's closed

#     def connect_to_websocket(connect_btc=True, connect_eth=True):
#         websocket.enableTrace(True)
#         stream_name_btc = "btcusdt@ticker"
#         if connect_btc:
#             ws_btc = websocket.WebSocketApp("wss://stream.binance.com:9443/ws/{}".format(stream_name_btc),
#                                             on_message=on_message_btc,
#                                             on_error=on_error,
#                                             on_close=on_close)
#             ws_btc.run_forever()

#         if connect_eth:
#             stream_name_eth = "ethusdt@ticker"
#             ws_eth = websocket.WebSocketApp("wss://stream.binance.com:9443/ws/{}".format(stream_name_eth),
#                                             on_message=on_message_eth,
#                                             on_error=on_error,
#                                             on_close=on_close)
#             ws_eth.run_forever()

#     def check_connection():
#         threading.Timer(3600, check_connection).start()  # check every 1 hour
#         current_time = datetime.datetime.now()
#         today_2pm_buffer = current_time.replace(
#             hour=15, minute=0, second=0, microsecond=0)
#         if current_time >= today_2pm_buffer and not on_message_btc.has_crossed_2pm:
#             connect_to_websocket(connect_eth=False)
#         if current_time >= today_2pm_buffer and not on_message_eth.has_crossed_2pm:
#             connect_to_websocket(connect_btc=False)

#     connect_to_websocket()  # initial connection to websocket
#     check_connection()  # start checking connection
