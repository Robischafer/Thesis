from preprocessing import stock_price, Market_Value, Total_return_index, stoxx600_price, eua_futures_price, \
    CO2_emissions_total
import pandas as pd
import numpy as np
from scipy.optimize import nnls, Bounds, minimize
import statsmodels.api as sm
import math
import matplotlib.pyplot as plt

# determine carbon footprint reduction
# graph carbon footprint
# determine tracking error

CO2_emissions_summary = CO2_emissions_total.sort_values(by=["2020-01-01"], axis=1)
CO2_emissions_mean = CO2_emissions_summary.mean(axis=1)
CO2_emissions_var = CO2_emissions_summary.var(axis=0)

stoxx600_top10_components = ["NESTLE 'N'", "ASML HOLDING", "ROCHE HOLDING", "LVMH", "ASTRAZENECA",
                             "SHELL", "NOVARTIS 'R'", "TOTALENERGIES", "L'OREAL", "RIO TINTO"]


# sorted[stoxx600_top10_components]


def moments(weight, df):
    mu = df.dot(weight.transpose())
    cov_matrix = df.cov()
    # variance portfolio
    sigma = weight.dot(cov_matrix.dot(weight.transpose()))

    return [mu, sigma]


def portfolio_valuation(df, weight):
    portfolio_value = df.dot(weight.transpose())
    return portfolio_value


def tracking_error_calc(pf, benchmark):
    return math.sqrt((pf - benchmark).var())


End_of_year = list()
for i in range(2012, 2021):
    temp = str(i) + "-12-31"
    End_of_year.append(temp)

Start_of_month = list()
for i in range(2012, 2021):
    for j in range(1, 12):
        temp = str(i)
        temp = temp + "-0" + str(j) + "-01"
        Start_of_month.append(temp)

global_start = "2011-12-30"
global_end = End_of_year[-1]

# indexing done based on 2012-01-01
# stoxx600_price
stoxx600_price_mean = stoxx600_price.mean()
stoxx600_price_var = stoxx600_price.var()
stoxx600_returns = stoxx600_price.pct_change(1)
stoxx600_returns_mean = stoxx600_returns.mean()
stoxx600_returns_var = stoxx600_returns.var()
# eua_futures_price
eua_futures_price_mean = eua_futures_price.mean()
eua_futures_price_var = eua_futures_price.var()
eua_futures_returns = eua_futures_price.pct_change(1)
eua_futures_returns_mean = eua_futures_returns.mean()
eua_futures_returns_var = eua_futures_returns.var()
# stock_price
stock_price_mean = stock_price.mean(axis=0)
stock_price_var = stock_price.var(axis=0)
stock_returns = stock_price.pct_change(1)
stock_returns_mean = stock_returns.mean()
stock_returns_var = stock_returns.var()

# per share participation (monthly)
Share_Monthly = stock_price[Market_Value.columns].groupby(pd.Grouper(freq="MS")).first()[0:-1] / Market_Value
Share_Yearly = stock_price[Market_Value.columns].groupby(pd.Grouper(freq="YS")).first()[0:-1] / Market_Value.groupby(
    pd.Grouper(freq="YS")).first()

# select data that is both in stock price, market value and CO2 emissions total
a = Share_Yearly[global_start:global_end]
b = CO2_emissions_total[global_start:global_end]
c = Market_Value[global_start:global_end]
d = Total_return_index[global_start:global_end]
Sample = set(a) & set(b) & set(c) & set(d)
# len(Sample)

Initial_Capital = 1000000
# emissions per unit of stock
h = a[Sample].multiply(b[Sample], axis=1)
eua_carbon_relation = -pd.DataFrame(np.ones(h.shape[0]))
eua_carbon_relation.index = h.index
carbon_participation_per_unit_stock = h.merge(eua_carbon_relation, how="inner", on="Date")
# tonne CO2 per EUR 1M invested
k = Initial_Capital / Market_Value[Sample].groupby(pd.Grouper(freq="YS")).first()[0:-2]
carbon_participation_per_million = k.multiply(CO2_emissions_total[Sample], axis=1)
l = - Initial_Capital / eua_futures_price[global_start:global_end].groupby(pd.Grouper(freq="YS")).first()
carbon_participation_per_million = carbon_participation_per_million.merge(l, how="inner", on="Date")

CO2_emissions_total_per_year = CO2_emissions_total.sum(axis=1)
# select 50 biggest market valuation
# i = Market_Value[Sample]["2012-01-01":"2012-01-02"].sort_values("2012-01-01", axis=1)
# biggest50 = i.iloc[:, -50:].columns

# plt.plot(g)
# plt.plot(i)

# perform KPSS test to test if the data stationary
sm.tsa.stattools.kpss(stoxx600_price, regression='ct')
sm.tsa.stattools.kpss(stoxx600_returns[1:-1], regression='ct')

sm.tsa.stattools.kpss(eua_futures_price, regression='ct')
sm.tsa.stattools.kpss(eua_futures_returns[1:-1], regression='ct')

# 1. created index matching stoxx 600, minimize tracking error
# 2. created same index + carbon hedge (EUA futures), minimize tracking error + track carbon reduction
# 3. same as 2. + carbon reduction constraints

########################################################################################################################
# 1. created index matching stoxx 600, minimize tracking error
# Non-negative Least Squares (NNLS) Optimization
# Lawson C., Hanson R.J., (1987) Solving Least Squares Problems, SIAM
"""
start = "2011-12-31"
end = "2016-12-31"

trainX = Total_return_index[Sample][start:end]

# taking stoxxx600 total return index and selecting the correct period, then indexing on period start (2012-01-01)
temp = stoxx600_price[:][start:end]
trainY = temp.div(temp.iloc[0]) * 100

result_base = nnls(trainX, trainY)
residuals_base = result_base[1]
leverage_factor_base = sum(result_base[0])
weights_base = result_base[0] / leverage_factor_base

Index_base = Total_return_index[Sample][global_start:global_end].mul(weights_base.transpose(), axis=1).sum(axis=1)
pf_emissions = CO2_emissions_total[Sample] * weights_base
pf_emissions_total = pf_emissions.sum(axis=1)
TE_base = tracking_error_calc(Index_base, stoxx600_price)

plot_index(Index_base, stoxx600_price[:][global_start:global_end], "Replication Stoxx600", "Index (base 2012-01-01)")

corr_base = Index_base.corr(stoxx600_price[:][global_start:global_end])
[Tstat_base, T_pvalue_base] = stats.ttest_ind(
    Index_base.pct_change(1)[global_start:global_end][1:-1],
    stoxx600_returns[global_start:global_end][1:-1], equal_var=False)
[Fstat_base, F_pvalue_base] = f_oneway(
    Index_base.pct_change(1)[global_start:global_end][1:-1],
    stoxx600_returns[global_start:global_end][1:-1])
"""
########################################################################################################################
# 2. created same index + carbon hedge (EUA futures), minimize tracking error + track carbon reduction

# Non-negative Least Squares (NNLS) Optimization
# Lawson C., Hanson R.J., (1987) Solving Least Squares Problems, SIAM

# Bounds (0, 1)
# sum(weight) = 1
# training period 2012-01-01 to 2012-13-31
start = "2012-01-01"
end = "2016-12-31"

# create index for eua futures based on 2012-01-01
eua_futures_index = eua_futures_price[global_start:global_end] / eua_futures_price[global_start:global_end][0] * 100
trainX = Total_return_index[Sample][start:end].merge(eua_futures_index[start:end], how="inner", on="Date")
TRI_extra = Total_return_index[Sample][global_start:global_end].merge(eua_futures_index[global_start:global_end],
                                                                      how="inner", on="Date")

# taking stoxxx600 total return index and selecting the correct period, then indexing on period start (2012-01-01)
temp = stoxx600_price[:][start:end]
trainY = temp.div(temp.iloc[0]) * 100

corr_no_constraints = TRI_extra.corr()

result_no_constraints = nnls(trainX, trainY)
residuals_no_constraints = result_no_constraints[1]
leverage_factor_no_constraints = sum(result_no_constraints[0])
weights_no_constraints = result_no_constraints[0] / leverage_factor_no_constraints

Index_no_constraints = TRI_extra.mul(weights_no_constraints.transpose(), axis=1).sum(axis=1)

TE_no_constraints = tracking_error_calc(Index_no_constraints, stoxx600_price)
# carbon emissions of portfolio depending on the initial capital, carbon emissions per shares and the allocation
pf_emissions_no_constraints = (carbon_participation_per_million * weights_no_constraints).sum(axis=1)

########################################################################################################################
# 3. created index with carbon reduction constraints, minimize tracking error + track carbon reduction

# training period 2012-01-01 to 2012-13-31
start = "2011-12-30"
end = "2016-12-31"

# create index for eua futures based on 2012-01-01
eua_futures_index = eua_futures_price[global_start:global_end] / eua_futures_price[global_start:global_end][0] * 100
Total_return_index_train = Total_return_index[Sample][start:end].merge(eua_futures_index[start:end], how="inner",
                                                                       on="Date")
TRI_extra = Total_return_index[Sample][global_start:global_end].merge(eua_futures_index[global_start:global_end],
                                                                      how="inner", on="Date")
stoxx600_TRI = stoxx600_price[:][global_start:global_end].div(stoxx600_price[:][global_start:global_end].iloc[1]) * 100

# trainX = Total_return_index_train.pct_change(1)[1:-1]
trainX = Total_return_index_train
v = (Total_return_index_train.pct_change(1)[1:-1]).mean(axis=0)
# taking stoxxx600 total return index and selecting the correct period, then indexing on period start (2012-01-01)
temp = stoxx600_price[:][start:end].div(stoxx600_price[:][start:end].iloc[0]) * 100
# trainY = temp.pct_change(1)[1:-1]
trainY = temp

cov = trainX.cov()
W = np.ones((v.shape[0], 1)) * (1.0 / v.shape[0])
shape = W.shape


def func(weight, df, target, shape):
    weight = weight.reshape(shape)
    return ((df.transpose() * weight).sum(axis=0) - target).var()


def optimize(func, weight, df, target, shape, eua_minimum):
    opt_bounds = Bounds(0, 1)
    opt_constraints = ({'type': 'eq', 'fun': lambda weight: 1.0 - np.sum(weight)},
                       {'type': 'ineq', 'fun': lambda weight: weight[-1] - eua_minimum}
                       )
    results = minimize(func, weight,
                       args=(df, target, shape),
                       method='SLSQP',
                       bounds=opt_bounds,
                       constraints=opt_constraints)
    return results


########################################################################################################################
# 30% EUA requirement

results_constraint_30p = optimize(func, W, trainX, trainY, shape, eua_minimum=0.3)
weights_constraint_30p = results_constraint_30p["x"]
Index_constraint_30p = (TRI_extra * weights_constraint_30p).sum(axis=1)

TE_post_ante_constraint_30p = tracking_error_calc(Index_constraint_30p, stoxx600_price)
TE_ex_ante_constraint_30p = tracking_error_calc(Index_constraint_30p[start:end], stoxx600_price[start:end])
# carbon emissions of portfolio depending on the initial capital, carbon emissions per shares and the allocation
pf_emissions_constraint_30p = (carbon_participation_per_million * weights_constraint_30p).sum(axis=1)

########################################################################################################################
# 20% EUA requirement

results_constraint_20p = optimize(func, W, trainX, trainY, shape, eua_minimum=0.2)
weights_constraint_20p = results_constraint_20p["x"]
Index_constraint_20p = (TRI_extra * weights_constraint_20p).sum(axis=1)

TE_post_ante_constraint_20p = tracking_error_calc(Index_constraint_20p, stoxx600_price)
TE_ex_ante_constraint_20p = tracking_error_calc(Index_constraint_20p[start:end], stoxx600_price[start:end])
# carbon emissions of portfolio depending on the initial capital, carbon emissions per shares and the allocation
pf_emissions_constraint_20p = (carbon_participation_per_million * weights_constraint_20p).sum(axis=1)

########################################################################################################################
# 10% EUA requirement

results_constraint_10p = optimize(func, W, trainX, trainY, shape, eua_minimum=0.1)
weights_constraint_10p = results_constraint_10p["x"]
Index_constraint_10p = (TRI_extra * weights_constraint_10p).sum(axis=1)

TE_post_ante_constraint_10p = tracking_error_calc(Index_constraint_10p, stoxx600_price)
TE_ex_ante_constraint_10p = tracking_error_calc(Index_constraint_10p[start:end], stoxx600_price[start:end])
# carbon emissions of portfolio depending on the initial capital, carbon emissions per shares and the allocation
pf_emissions_constraint_10p = (carbon_participation_per_million * weights_constraint_10p).sum(axis=1)

########################################################################################################################
# 5% EUA requirement

results_constraint_5p = optimize(func, W, trainX, trainY, shape, eua_minimum=0.05)
weights_constraint_5p = results_constraint_5p["x"]
Index_constraint_5p = (TRI_extra * weights_constraint_5p).sum(axis=1)

TE_post_ante_constraint_5p = tracking_error_calc(Index_constraint_5p, stoxx600_price)
TE_ex_ante_constraint_5p = tracking_error_calc(Index_constraint_5p[start:end], stoxx600_price[start:end])
# carbon emissions of portfolio depending on the initial capital, carbon emissions per shares and the allocation
pf_emissions_constraint_5p = (carbon_participation_per_million * weights_constraint_5p).sum(axis=1)

########################################################################################################################
# 1% EUA requirement

results_constraint_1p = optimize(func, W, trainX, trainY, shape, eua_minimum=0.01)
weights_constraint_1p = results_constraint_1p["x"]
Index_constraint_1p = (TRI_extra * weights_constraint_1p).sum(axis=1)

TE_post_ante_constraint_1p = tracking_error_calc(Index_constraint_1p, stoxx600_price)
TE_ex_ante_constraint_1p = tracking_error_calc(Index_constraint_1p[start:end], stoxx600_price[start:end])
# carbon emissions of portfolio depending on the initial capital, carbon emissions per shares and the allocation
pf_emissions_constraint_1p = (carbon_participation_per_million * weights_constraint_1p).sum(axis=1)

########################################################################################################################
# compare with nnls optimization
# 0% EUA requirement

results_constraint_0p = optimize(func, W, trainX, trainY, shape, eua_minimum=0)
weights_constraint_0p = results_constraint_0p["x"]
Index_constraint_0p = (TRI_extra * weights_constraint_0p).sum(axis=1)

TE_post_ante_constraint_0p = tracking_error_calc(Index_constraint_0p, stoxx600_price)
TE_ex_ante_constraint_0p = tracking_error_calc(Index_constraint_0p[start:end], stoxx600_price[start:end])
# carbon emissions of portfolio depending on the initial capital, carbon emissions per shares and the allocation
pf_emissions_constraint_0p = (carbon_participation_per_million * weights_constraint_0p).sum(axis=1)

########################################################################################################################
# comparison

reduction_CO2_10p = (pf_emissions_constraint_10p - pf_emissions_constraint_0p) / pf_emissions_constraint_0p
reduction_CO2_5p = (pf_emissions_constraint_5p - pf_emissions_constraint_0p) / pf_emissions_constraint_0p
reduction_CO2_1p = (pf_emissions_constraint_1p - pf_emissions_constraint_0p) / pf_emissions_constraint_0p
reduction_CO2_0p = (pf_emissions_no_constraints - pf_emissions_constraint_0p) / pf_emissions_constraint_0p

TE_post_ante = [TE_post_ante_constraint_0p, TE_post_ante_constraint_1p, TE_post_ante_constraint_5p,
                TE_post_ante_constraint_10p
    , TE_post_ante_constraint_20p, TE_post_ante_constraint_30p
                ]
TE_ex_ante = [TE_ex_ante_constraint_0p, TE_ex_ante_constraint_1p, TE_ex_ante_constraint_5p, TE_ex_ante_constraint_10p
    , TE_ex_ante_constraint_20p, TE_ex_ante_constraint_30p
              ]

pf_emissions = [pf_emissions_constraint_0p, pf_emissions_constraint_1p, pf_emissions_constraint_5p,
                pf_emissions_constraint_10p, pf_emissions_constraint_20p, pf_emissions_constraint_30p]


########################################################################################################################


def market_weighting(market_cap):
    """
    :param market_cap: monthly market cap pandas DataFrame
    :return: portfolio weights
    """
    market_cap_total = market_cap.sum(axis=1)
    weight = market_cap.mul(1 / market_cap_total, axis=0)

    return weight


def portfolio_valuation_monthly(df, weight):
    """
    :param df:
    :param weight:
    :return:
    """
    df["date"] = df.index
    temp = df.groupby(pd.Grouper(freq="MS")).first()
    temp = temp.drop("date", axis=1)

    portfolio_value = temp.dot(weight.transpose())
    return portfolio_value


df_market = stock_price[Sample][global_start:global_end]
market_weight = market_weighting(Market_Value[Sample])

# only take columns from stock price that have market capitalization data

df_market = df_market[market_weight.columns]

portfolio_value_market = portfolio_valuation_monthly(df_market, market_weight[:][global_start:global_end])

# plt.plot(portfolio_value_market.index, np.diag(portfolio_value_market))

"""
equal weight 
"""


def equal_weighting(df):
    weight = np.ones(shape=(1, df.shape[1])) / df.shape[1]
    return weight


Sample_extra = list(Sample)
Sample_extra.append("ICE-ECX CER Daily Futures")

equal_weight = equal_weighting(TRI_extra[Sample_extra][global_start:global_end])
portfolio_value_equal = portfolio_valuation(TRI_extra[Sample_extra][global_start:global_end], equal_weight)
pf_emissions_equal_weight = (carbon_participation_per_million * equal_weight).sum(axis=1)

########################################################################################################################

"""
minimum variance

covariance_matrix = [[s1 ^ 2, s1 * s2],
                     [s2 * s1, s2 ^ 2]]
minimum_variance_weights = [w1, w2]

portfolio_variance = weights * covariance_matrix * weights
"""

# training period 2012-01-01 to 2012-13-31
start = "2011-12-30"
end = "2016-12-31"

# create index for eua futures based on 2012-01-01
eua_futures_index = eua_futures_price[global_start:global_end] / eua_futures_price[global_start:global_end][0] * 100
Total_return_index_train = Total_return_index[Sample][start:end].merge(eua_futures_index[start:end], how="inner",
                                                                       on="Date")
TRI_extra = Total_return_index[Sample][global_start:global_end].merge(eua_futures_index[global_start:global_end],
                                                                      how="inner", on="Date")

trainX = Total_return_index_train.pct_change(1)[1:-1]
v = (Total_return_index_train.pct_change(1)[1:-1]).mean(axis=0)
# taking stoxxx600 total return index and selecting the correct period, then indexing on period start (2012-01-01)
temp = stoxx600_price[:][start:end].div(stoxx600_price[:][start:end].iloc[0]) * 100
trainY = temp.pct_change(1)[1:-1]

cov = trainX.cov()
# initial guess for weight
W0 = np.ones((v.shape[0], 1)) * (1.0 / v.shape[0])


# Function to optimize
def ret_risk(W, exp_ret, cov):
    return -((W.T @ exp_ret) / (W.T @ cov @ W) ** 0.5)


# Function that runs optimizer
def optimize_minvar(func, W, exp_ret, cov):
    opt_bounds = Bounds(0, 1)
    opt_constraints = ({'type': 'eq',
                        'fun': lambda W: 1.0 - np.sum(W)}
    )
    optimal_weights = minimize(func, W,
                               args=(exp_ret, cov),
                               method='SLSQP',
                               bounds=opt_bounds,
                               constraints=opt_constraints)
    return optimal_weights['x']


weight_minimum_variance = optimize_minvar(ret_risk, W0, v, cov)
portfolio_value_minvar = (TRI_extra * weight_minimum_variance).sum(axis=1)
pf_emissions_minvar = (carbon_participation_per_million * weight_minimum_variance).sum(axis=1)

"""
possible addition 

most diversification

cap weight
"""
