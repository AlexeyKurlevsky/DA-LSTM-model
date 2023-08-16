import click
import pandas as pd
import numpy as np
import logging


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def calculate_hdb(input_path: str, output_path: str) -> None:
    """
    Function to calculate characteristic of dynamic balance
    :param input_path: Dataframe
    :param output_path:
    """
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv(input_path, parse_dates=["Дата"])
    logging.info("Data read")
    n = df.shape[0]
    hdb = np.zeros(n)
    for index_recover in range(n):
        for index_noticed in range(index_recover, 0, -1):
            if (
                df["Выздоровевших и умерших"].iloc[index_recover]
                <= df["Выявлено всего"].iloc[index_noticed]
            ):
                hdb[index_recover] = index_recover - index_noticed

    df["Характерстика ДБ"] = hdb
    df.to_csv(output_path, index=False)
    logging.info("HDB calculated")


if __name__ == "__main__":
    calculate_hdb()
