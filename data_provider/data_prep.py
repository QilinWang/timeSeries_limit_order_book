import datetime
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import datatable as dt

# from sklearn.preprocessing import FunctionTransformer

### CSV -> data in resampled time frequency

def day_csv_transform(
        df,
        str_col="Date-Time",
        freq="5S",
):

    df.rename(columns={str_col: "time"}, inplace=True)


    df["time"] = pd.to_datetime(df["time"])  # .dt.tz_localize(None)

    select_cols = ["time"]
    price_cols = []
    size_cols = []

    for k in range(1, 11):
        price_cols += [f"L{k}-AskPrice", f"L{k}-BidPrice"]
        size_cols += [f"L{k}-AskSize", f"L{k}-BidSize"]
        select_cols += [f'L{k}-AskPrice, L{k}-BidPrice, L{k}-AskSize, L{k}-BidSize']
    df = df[select_cols]
    df = df.dropna()
    df.reset_index(inplace=True)

    group_df = df.groupby(pd.Grouper(key="time", freq=freq))

    lst_price_ask = []
    lst_size_ask = []
    lst_price_buy = []
    lst_size_buy = []

    def ask_price(level):
        return lambda x: np.mean(np.repeat(x[f"L{level}-AskPrice"], x[f"L{level}-AskSize"]))


    def ask_size(level):
        return lambda x: np.mean(x[f"L{level}-AskSize"])


    def bid_price(level):
        return lambda x: np.mean(np.repeat(x[f"L{level}-BidPrice"], x[f"L{level}-BidSize"]))


    def bid_size(level):
        return lambda x: np.mean(x[f"L{level}-BidSize"])

    for i in range(10, 0, -1):

        _price = group_df.apply(ask_price(i)).transform(lambda x: x.fillna(0))
        lst_price_ask.append(_price)
        _size = group_df.apply(ask_size(i)).transform(lambda x: x.fillna(0))
        lst_size_ask.append(_size)

    for i in range(1, 11):

        _price = group_df.apply(bid_price(i)).transform(lambda x: x.fillna(0))
        lst_price_buy.append(_price)
        _size = group_df.apply(bid_size(i)).transform(lambda x: x.fillna(0))
        lst_size_buy.append(_size)

    lst_price = lst_price_ask + lst_price_buy
    lst_size = lst_size_ask + lst_size_buy

    p_df = pd.concat(lst_price, axis=1, keys=price_cols)
    v_df = pd.concat(lst_size, axis=1, keys=size_cols)

    p_df.reset_index(inplace=True)
    v_df.reset_index(inplace=True)

    time_features = get_time_features(p_df)

    macro_midprice = pd.DataFrame(
        (
                p_df["L1-AskPrice"] * v_df["L1-AskSize"] + p_df["L1-BidPrice"] * v_df["L1-BidSize"]
        ) / (v_df["L1-AskSize"] + v_df["L1-BidSize"]),
        columns=["midprice"],
        ).fillna(0)

    total = pd.concat(
        [
            p_df.drop(["time"], axis=1),
            v_df.drop(["time"], axis=1),
            time_features,
            macro_midprice,
        ],
        axis=1,
    )

    return total


def encode(data, col, max_val, time_col="time"):
    data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)
    return data


def encode_withSeries(series, max_val, set_name_to):
    a = np.sin(2 * np.pi * series / max_val)
    b = np.cos(2 * np.pi * series / max_val)
    return pd.concat([series, a, b], axis=1, keys=[set_name_to, set_name_to + "_sin", set_name_to + "_cos"],)


def get_time_features(df, time_col="time", coef=10**3):
    # ! these are for grouped subjuects

    ts_micro = (df["time"].values.astype(np.int64) // coef)  # nanosec to microsec, divided by 10**3
    ts_micro = pd.Series(ts_micro)

    microseconds_in_day = 24 * 60 * 60 * 1e6

    s1 = encode_withSeries(df["time"].dt.month, 12, "month")
    s2 = encode_withSeries(df["time"].dt.day_of_year, df["time"].dt.is_leap_year.astype(int) + 365, "day")
    s3 = encode_withSeries(ts_micro, microseconds_in_day, "microseconds")
    s4 = encode_withSeries(df["time"].dt.day_of_week, 6, "day_of_week")

    return pd.concat([s1, s2, s3, s4], axis=1)

def read_parquet(folder_path):
    train_df = pd.read_parquet(f"{folder_path}/train_df.parquet")
    valid_df = pd.read_parquet(f"{folder_path}/valid_df.parquet")
    test_df = pd.read_parquet(f"{folder_path}/test_df.parquet")
    return train_df, valid_df, test_df


def split_parquet(folder_path, file_name, train_ratio=0.7, valid_ratio=0.2, save=True):
    df = pd.read_parquet(f"{folder_path}/{file_name}")
    n = len(df)
    train_df = df[0 : int(n * train_ratio)]
    valid_df = df[int(n * train_ratio) : int(n * (train_ratio + valid_ratio))]
    test_df = df[int(n * (train_ratio + valid_ratio)) :]
    if save:
        train_df.to_parquet(f"{folder_path}/train_df.parquet")
        valid_df.to_parquet(f"{folder_path}/valid_df.parquet")
        test_df.to_parquet(f"{folder_path}/test_df.parquet")
    return train_df, valid_df, test_df


def data_classification(X, Y, T):

    # * Eg: N = 200, D = 40, T = 5, then in total we have 196 windows (N-T+1);
    # for each window, time length is 5 and level length is 40

    dataY = Y[T - 1 : N]  # Get Y[4:200]

    [N, D] = X.shape
    dataX = np.zeros((N - T + 1, T, D))
    for t in range(T, N + 1):  # * range([5, 201])
        dataX[t - T] = X[
            t - T : t, :
        ]  # data[25-5] = X[25-5:25] ie data[20] = X[20:25] which is T by D

    # * end_point is 5,      index of dataX is 0,         corr.indices of X is timestep range(0:5), i.e 0-4
    # * ...
    # * end_point is 200,    index of dataX is 195,       corr.indices of X is timestep range(195:200), i.e 195-200

    return dataX, dataY


def seq_break_into_intervals(X, T):  # [N, d] -> [N - T + 1, T, D]
    # 0-5 TO 0, 1-6 TO 1, ... each seq is T length window of starting point
    # TODO what is the better way to divide these?
    [N, D] = X.shape
    dataX = np.zeros((N - T + 1, T, D))
    for i in range(0, N - T + 1):  # 0-5 to 0, 1-6 to 1 ... etc
        dataX[i] = X[i : i + T, :]
    return dataX


# def sin_transformer(period):
# 	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

# def cos_transformer(period):
# 	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))














def make_parquet(  # 2021-2-10 to 2022-03-25
    begin_year,
    begin_month,
    begin_day,
    end_year,
    end_month,
    end_day,
    folder="./data",
    save_folder=".data_new",
    ticker="SB",
):

    os.makedirs(f"{folder}/{ticker}/npy/", exist_ok=True)

    cols = ["Date-Time"]
    for k in range(1, 11):
        cols += [
            "L%s-AskPrice" % k,
            "L%s-AskSize" % k,
            "L%s-BidPrice" % k,
            "L%s-BidSize" % k,
        ]

    file_list = []
    for file in os.listdir(f"{folder}/{ticker}/raw/"):
        year, month, day = map(
            lambda x: int(x),
            file.split(sep="_", maxsplit=-1)[1].split(sep=".")[0].split("-"),
        )
        currentDateTime = datetime.datetime(year, month, day)
        file_list.append(currentDateTime)

    # running_queue = deque([]) # 5 days of running average

    df_list = []

    for i, dt in enumerate(file_list):
        if (datetime.datetime(begin_year, begin_month, begin_day) <= dt) and (
            dt <= datetime.datetime(end_year, end_month, end_day)
        ):

            d = pd.read_csv(
                f"{folder}/{ticker}/raw/{ticker}_{dt.year}-{dt.month:02d}-{dt.day:02d}.csv"
            )

            print(f"The {i}th file for {dt} has length {len(d)}\n")

            if len(d) < 1000:
                print(
                    "The {i}th document for {dt} has few observation and will be skipped\n"
                )
                continue

            total = day_csv_transform(d)

    df_list.append(total)

    df = pd.concat(df_list, axis=0)
    os.makedirs(
        f"{save_folder}/{ticker}_{begin_year}{begin_month:02d}{begin_day:02d}_{end_year}{end_month:02d}{end_day:02d}",
        exist_ok=True,
    )
    df.to_parquet(
        f"{save_folder}/{ticker}_{begin_year}{begin_month:02d}{begin_day:02d}_{end_year}{end_month:02d}{end_day:02d}/df.parquet"
    )

    return df


def make_parquet(  # 2021-2-10 to 2022-03-25
    begin_year,
    begin_month,
    begin_day,
    end_year,
    end_month,
    end_day,
    folder="./data",
    save_folder=".data_new",
    ticker="SB",
):

    cols = ["Date-Time"]
    for k in range(1, 11):
        cols += [
            f"L{k}-AskPrice",
            f"L{k}-AskSize",
            f"L{k}-BidPrice",
            f"L{k}-BidSize",
        ]

    file_list = []
    for file in os.listdir(f"{folder}/{ticker}/raw/"):
        year, month, day = map(
            lambda x: int(x),
            file.split(sep="_", maxsplit=-1)[1].split(sep=".")[0].split("-"),
        )
        currentDateTime = datetime.datetime(year, month, day)
        file_list.append(currentDateTime)

    # running_queue = deque([]) # 5 days of running average

    df_list = []

    for i, dt in enumerate(file_list):
        if (datetime.datetime(begin_year, begin_month, begin_day) <= dt) and (
            dt <= datetime.datetime(end_year, end_month, end_day)
        ):

            d = pd.read_csv(
                f"{folder}/{ticker}/raw/{ticker}_{dt.year}-{dt.month:02d}-{dt.day:02d}.csv"
            )

            print(f"The {i}th file for {dt} has length {len(d)}\n")

            if len(d) < 1000:
                print(
                    f"The {i}th document for {dt} has few observation and will be skipped\n"
                )
                continue

            total = day_csv_transform(d)

            df_list.append(total)

    df = pd.concat(df_list, axis=0)
    os.makedirs(
        f"{save_folder}/{ticker}_{begin_year}{begin_month:02d}{begin_day:02d}_{end_year}{end_month:02d}{end_day:02d}",
        exist_ok=True,
    )
    df.to_parquet(
        f"{save_folder}/{ticker}_{begin_year}{begin_month:02d}{begin_day:02d}_{end_year}{end_month:02d}{end_day:02d}/df.parquet"
    )

    return df
