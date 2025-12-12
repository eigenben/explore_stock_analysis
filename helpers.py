import os
import pandas as pd

def prepare_data_features(df):
    df_new = pd.DataFrame()
    df_new["open"] = df["Open"]
    df_new["open_1"] = df["Open"].shift(1)
    df_new["close_1"] = df["Close"].shift(1)
    df_new["high_1"] = df["High"].shift(1)
    df_new["low_1"] = df["Low"].shift(1)
    df_new["volume_1"] = df["Volume"].shift(1)
    df_new["avg_price_5"] = df["Close"].rolling(5).mean().shift(1)
    df_new["avg_price_30"] = df["Close"].rolling(21).mean().shift(1)
    df_new["avg_price_365"] = df["Close"].rolling(252).mean().shift(1)
    df_new["ratio_avg_price_5_30"] = df_new["avg_price_5"] / df_new["avg_price_30"]
    df_new["ratio_avg_price_5_365"] = df_new["avg_price_5"] / df_new["avg_price_365"]
    df_new["ratio_avg_price_30_365"] = df_new["avg_price_30"] / df_new["avg_price_365"]
    df_new["avg_volume_5"] = df["Volume"].rolling(5).mean().shift(1)
    df_new["avg_volume_30"] = df["Volume"].rolling(21).mean().shift(1)
    df_new["avg_volume_365"] = df["Volume"].rolling(252).mean().shift(1)
    df_new["ratio_avg_volume_5_30"] = df_new["avg_volume_5"] / df_new["avg_volume_30"]
    df_new["ratio_avg_volume_5_365"] = df_new["avg_volume_5"] / df_new["avg_volume_365"]
    df_new["ratio_avg_volume_30_365"] = (
        df_new["avg_volume_30"] / df_new["avg_volume_365"]
    )
    df_new["std_price_5"] = df["Close"].rolling(5).std().shift(1)
    df_new["std_price_30"] = df["Close"].rolling(21).std().shift(1)
    df_new["std_price_365"] = df["Close"].rolling(252).std().shift(1)
    df_new["ratio_std_price_5_30"] = df_new["std_price_5"] / df_new["std_price_30"]
    df_new["ratio_std_price_5_365"] = df_new["std_price_5"] / df_new["std_price_365"]
    df_new["ratio_std_price_30_365"] = df_new["std_price_30"] / df_new["std_price_365"]
    df_new["std_volume_5"] = df["Volume"].rolling(5).std().shift(1)
    df_new["std_volume_30"] = df["Volume"].rolling(21).std().shift(1)
    df_new["std_volume_365"] = df["Volume"].rolling(252).std().shift(1)
    df_new["ratio_std_volume_5_30"] = df_new["std_volume_5"] / df_new["std_volume_30"]
    df_new["ratio_std_volume_5_365"] = df_new["std_volume_5"] / df_new["std_volume_365"]
    df_new["ratio_std_volume_30_365"] = (
        df_new["std_volume_30"] / df_new["std_volume_365"]
    )
    df_new["return_1"] = (
        (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)
    ).shift(1)
    df_new["return_5"] = (
        (df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5)
    ).shift(1)
    df_new["return_30"] = (
        (df["Close"] - df["Close"].shift(21)) / df["Close"].shift(21)
    ).shift(1)
    df_new["return_365"] = (
        (df["Close"] - df["Close"].shift(252)) / df["Close"].shift(252)
    ).shift(1)
    df_new["moving_avg_5"] = df_new["return_1"].rolling(5).mean().shift(1)
    df_new["moving_avg_30"] = df_new["return_1"].rolling(21).mean().shift(1)
    df_new["moving_avg_365"] = df_new["return_1"].rolling(252).mean().shift(1)
    df_new["close"] = df["Close"]
    return df_new

def load_stock_data(filename):
    path = os.path.join("data", f"{filename}.csv")
    return prepare_data_features(pd.read_csv(path, index_col="Date"))
