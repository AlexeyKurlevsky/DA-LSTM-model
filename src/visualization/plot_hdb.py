import click

def plot_hdb() -> None:
    mpl.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(df_search.index, df_search['Характерстика ДБ'], label='Характерстика ДБ', color='cornflowerblue')
    ax1 = ax.twinx()
    ax1.plot(df_search.index, df_search['Заражений за день'], label='Заражений за день', color='salmon')
    ax.grid()
    ax.set_ylabel('Характерстика ДБ')
    ax1.set_ylabel('Заражений за день')
    ax.legend(loc='upper left')
    ax1.legend(loc='upper right')
    plt.show()