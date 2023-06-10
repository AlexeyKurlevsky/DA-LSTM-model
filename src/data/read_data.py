import click
import pandas as pd


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def read_data(input_path: str, output_path: str) -> None:
    """
    Prepare data for analysis
    :param input_path: path containing raw data
    :param output_path: first handle data
    """
    df = pd.read_csv(input_path, sep=";")
    df.reset_index(drop=True, inplace=True)
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%Y-%m-%d %H:%M:%S")
    df.rename(columns={"DateTime": "Дата", "Заражений": "Выявлено всего"}, inplace=True)

    df["% Выздоровлений"] = df["Выздоровлений"].pct_change()
    df["% Смертей"] = df["Смертей"].pct_change()
    df["% прирост"] = df["Выявлено всего"].pct_change()
    df.rename(columns={"DateTime": "Дата"}, inplace=True)
    df["Выздоровевших и умерших"] = df["Выздоровлений"] + df["Смертей"]
    df["Выздоровевших и умерших"] = df["Выздоровевших и умерших"]

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    read_data()
