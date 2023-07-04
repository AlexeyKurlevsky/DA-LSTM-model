import logging
import click
import pandas as pd
import numpy as np
import yaml


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def get_freq_components(input_path: str, output_path: str) -> None:
    logging.basicConfig(level=logging.INFO)

    params = yaml.safe_load(open("params.yaml"))["data_feature"]

    df = pd.read_csv(input_path, parse_dates=["Дата"])
    logging.info("Data read")

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

    day = 24 * 60 * 60 * params["day_freq"]

    df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))

    df_search = df[
        ["Дата", "Заражений за день", "Day sin", "Day cos", "Характерстика ДБ"]
    ]
    df_search.dropna(inplace=True)
    df_search.to_csv(output_path, index=False)
    logging.info("Frequency response calculated")


if __name__ == "__main__":
    get_freq_components()
