import matplotlib.pyplot as plt
from main import Index_no_constraints, Index_constraint_1p,  Index_constraint_5p, Index_constraint_10p, \
    stoxx600_TRI, global_start, global_end, portfolio_value_equal, portfolio_value_minvar, eua_futures_price
from main import pf_emissions_minvar, pf_emissions_equal_weight, pf_emissions_constraint_1p, pf_emissions_no_constraints


def plot_index(Index, Base, title, subtitle):
    plt.plot(Index, color="k", linestyle='solid', label="Replicated Index total return index")
    plt.plot(Base.index, Base, color='b', label='Stoxx600 total return index')

    plt.xlabel("Date")
    plt.ylabel("Index (base 2012-01-01)")
    plt.title(subtitle)
    plt.suptitle(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(title + ".png", bbox_inches='tight')
    plt.close()
    return 0


def plot_rolling_mean_var(df):
    rolling_mean = df.rolling(window=52).mean()
    rolling_var = df.rolling(window=52).var()

    plt.plot(df, color="b", label="data")
    plt.title("rolling average and rolling variance")
    plt.plot(rolling_mean, color="g", label="Rolling Mean")
    plt.plot(rolling_var, color="y", label="Rolling Var")
    plt.show()
    return 0


df = eua_futures_price
rolling_mean = df.rolling(window=5).mean()
rolling_var = df.rolling(window=5).var()

plt.title("EUA futures rolling mean")
plt.plot(rolling_mean, color="y", label="Rolling Mean")
plt.show()

plot_index(Index_no_constraints, stoxx600_TRI,
           "Replication Stoxx600", "no constraints")

plot_index(Index_constraint_1p, stoxx600_TRI,
           "Replication Stoxx600 with minimum 1% EUA futures", "")

plot_index(Index_constraint_5p, stoxx600_TRI,
           "Replication Stoxx600 with minimum 5% EUA futures", "")

plot_index(Index_constraint_10p, stoxx600_TRI,
           "Replication Stoxx600 with minimum 10% EUA futures", "")

# plot_index(Index_constraint_20p, stoxx600_TRI,
#           "Replication Stoxx600 with minimum 20% EUA futures", "")

# plot_index(Index_constraint_30p, stoxx600_TRI,
#           "Replication Stoxx600 with minimum 30% EUA futures", "")

plot_index(portfolio_value_equal, stoxx600_TRI, "Portfolio with equal weight", "")

plot_index(portfolio_value_minvar, stoxx600_TRI, "Minimum Variance Portfolio", "")


def carbon_footprint_plot(pf1, label1, pf2, label2, title):
    plt.plot(pf1, color='k', label=label1)
    plt.plot(pf2, color='b', label=label2)

    plt.xlabel("Date")
    plt.ylabel("CO2 emissions per EUR 1 million invested")
    plt.title("")
    plt.suptitle(title)
    plt.legend(loc='lower left', bbox_to_anchor=(0.7, 0.0))
    plt.grid(True)
    #plt.savefig(title + ".png", bbox_inches='tight')
    #plt.close()


carbon_footprint_plot(pf_emissions_constraint_1p, 'replication stoxx600 with 1% EUA minimum',
                      pf_emissions_no_constraints, 'replication stoxx600 with no constraint',
                      "Index carbon footprint")

carbon_footprint_plot(pf_emissions_minvar, 'minimum variance portfolio',
                      pf_emissions_equal_weight, 'equal weight portfolio',
                      "Relative index carbon footprint")

# plot_rolling_mean_var(stoxx600_price)
# plot_rolling_mean_var(eua_futures_price)
