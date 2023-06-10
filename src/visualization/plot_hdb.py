import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def plot_hdb(input_path: str, output_path: str) -> None:
    """
    Plot data for analysis
    :param input_path: path processed data
    :param output_path: save plot path
    """
    mpl.rcParams.update({"font.size": 14})
    df_search = pd.read_csv(input_path, parse_dates=["Дата"])
    df_search = df_search.set_index("Дата")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(
        df_search.index,
        df_search["Характерстика ДБ"],
        label="Характерстика ДБ",
        color="cornflowerblue",
    )
    ax1 = ax.twinx()
    ax1.plot(
        df_search.index,
        df_search["Заражений за день"],
        label="Заражений за день",
        color="salmon",
    )
    ax.grid()
    ax.set_ylabel("Характерстика ДБ")
    ax1.set_ylabel("Заражений за день")
    ax.legend(loc="upper left")
    ax1.legend(loc="upper right")
    plt.savefig(output_path)


if __name__ == "__main__":
    plot_hdb()
