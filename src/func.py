import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def get_hdb(df: pd.DataFrame) -> pd.Series:
    n = df.shape[0]
    hdb = np.zeros(n)
    for index_recover in range(n):
        for index_noticed in range(index_recover, 0, -1):
            if (
                df["Выздоровевших и умерших"].iloc[index_recover]
                <= df["Выявлено всего"].iloc[index_noticed]
            ):
                hdb[index_recover] = index_recover - index_noticed
    return hdb


def get_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.reset_index(drop=True, inplace=True)
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%Y-%m-%d %H:%M:%S")
    df.rename(columns={"DateTime": "Дата", "Заражений": "Выявлено всего"}, inplace=True)

    df["% Выздоровлений"] = df["Выздоровлений"].pct_change()
    df["% Смертей"] = df["Смертей"].pct_change()
    df["% прирост"] = df["Выявлено всего"].pct_change()
    df.rename(columns={"DateTime": "Дата"}, inplace=True)
    df["Выздоровевших и умерших"] = df["Выздоровлений"] + df["Смертей"]
    df["Выздоровевших и умерших"] = df["Выздоровевших и умерших"]

    df["Характерстика ДБ"] = get_hdb(df)

    df = df[
        [
            "Дата",
            "Выявлено всего",
            "Выздоровевших и умерших",
            "Заражений за день",
            "% прирост",
            "Выздоровлений",
            "Выздоровлений за день",
            "% Выздоровлений",
            "Смертей",
            "Смертей за день",
            "% Смертей",
            "Летальность, %",
            "Характерстика ДБ",
        ]
    ]

    df = df.drop(index=df[df["% прирост"] > 1].index)
    df.dropna(inplace=True)

    timestamp_s = df["Дата"].map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60 * 210

    df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))

    return df
