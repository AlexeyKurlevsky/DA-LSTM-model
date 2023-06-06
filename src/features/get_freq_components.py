import click
import pandas as pd
import numpy as np


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def get_freq_components(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path, parse_dates=["Дата"])
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

    df_search = df[["Дата", "Заражений за день", "Day sin", "Day cos", "Характерстика ДБ"]]
    df_search.dropna(inplace=True)
    df_search.to_csv(output_path, index=False)


if __name__ == '__main__':
    get_freq_components()
