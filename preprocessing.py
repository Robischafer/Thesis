import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# company = pd.read_excel()

# yearly data
carbon_emissions_all = pd.read_excel(io="Data raw/carbon_esg.xlsx", sheet_name="carbon emissions", header=0, index_col=0)
esg_score = pd.read_excel(io="Data raw/carbon_esg.xlsx", sheet_name="esg score co2", header=0, index_col=0)

Start_of_year = list()
for i in range(2012, 2021):
    temp = str(i) + "-01-01"
    Start_of_year.append(temp)

# in tonne of C02 emissions
CO2_emission_scope_1 = carbon_emissions_all.loc[:, 2012:2020]
CO2_emission_scope_1 = CO2_emission_scope_1.dropna()
CO2_emission_scope_1 = CO2_emission_scope_1.transpose()
CO2_emission_scope_1 = CO2_emission_scope_1.rename(
    columns=lambda x: x[0:-len("- CO2 Equivalents Emission Direct")-1], inplace=False
)
CO2_emission_scope_1.index = Start_of_year
CO2_emission_scope_1.index.name = "Date"
CO2_emission_scope_1.columns.name = "Direct emissions"

CO2_emission_scope_2 = carbon_emissions_all.loc[:, "2012.1":"2020.1"]
CO2_emission_scope_2 = CO2_emission_scope_2.dropna()
CO2_emission_scope_2 = CO2_emission_scope_2.transpose()
CO2_emission_scope_2 = CO2_emission_scope_2.rename(
    columns=lambda x: x[0:-len("- CO2 Equivalents Emission Direct")-1], inplace=False
)
CO2_emission_scope_2.index = Start_of_year
CO2_emission_scope_2.index.name = "Date"
CO2_emission_scope_2.columns.name = "Indirect emissions"

CO2_emission_scope_3 = carbon_emissions_all.loc[:, "2012.2":"2020.2"]
CO2_emission_scope_3 = CO2_emission_scope_3.dropna()
CO2_emission_scope_3 = CO2_emission_scope_3.transpose()
CO2_emission_scope_3 = CO2_emission_scope_3.rename(
    columns=lambda x: x[0:-len("- CO2 Equivalents Emission Direct")-1], inplace=False
)
CO2_emission_scope_3.index = Start_of_year
CO2_emission_scope_3.index.name = "Date"
CO2_emission_scope_3.columns.name = "Value chain emissions"

a = CO2_emission_scope_1.reset_index("Date").join(
    CO2_emission_scope_2.reset_index("Date"), lsuffix=" direct", rsuffix=" indirect"
)

CO2_emissions_total = CO2_emission_scope_1.add(CO2_emission_scope_2, fill_value=np.nan)
CO2_emissions_total = CO2_emissions_total.dropna(axis=1)
CO2_emissions_total.index = pd.to_datetime(CO2_emissions_total.index)


def plot_carbon():
    sum1 = CO2_emission_scope_1.sum(axis=1)
    sum2 = CO2_emission_scope_2.sum(axis=1)
    sum3 = sum1 + sum2

    plt.plot(sum1.index, sum1, color="r", label="Scope 1 emissions")
    plt.plot(sum2.index, sum2, color="g", label="Scope 2 emissions")
    plt.plot(sum3.index, sum3, color="y", label="Combined")

    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("million of CO2 tonne")
    plt.title("Scope 1 and Scope 2 carbon emissions")
    plt.grid(True)


# net sales/revenues
# sales = pd.read_excel(io="carbon_esg.xlsx", sheet_name="sales", header=0)
# number of employees, full time equivalent
# employee = pd.read_excel(io="carbon_esg.xlsx", sheet_name="employee", header=0)
# electricity produced/purchased
# electricity = []

# daily data

date = []
# closing price in EUR
stock_price = pd.read_excel(io="Data raw/Stock price.xlsx", sheet_name="Stock Price EUR",
                            header=1, index_col="Date")
stock_price = stock_price.dropna(axis=1)
stock_price = stock_price.rename(columns=lambda x: x[0:-6], inplace=False)

# Total return index formula
# RI_t = RI_t - 1 * PIt / PIt - 1 * (1 + DYt / (100 * N))

dividend_yield = pd.read_excel(io="Data raw/Stock price.xlsx", sheet_name="Dividend yield %",
                               header=0, index_col="Date")
Total_return_index = pd.read_excel(io="Data raw/Stock price.xlsx", sheet_name="Total return index",
                                   header=0, index_col="Date")

# drop NaN and normalized the data based on the first value of the year 2012 (2012-01-02)
Total_return_index = Total_return_index.dropna(axis=1)
Total_return_index = Total_return_index.div(Total_return_index.iloc[1], axis='columns') * 100
Total_return_index = Total_return_index.rename(columns=lambda x: x[0:-17], inplace=False)

# monthly market capitalization of corresponding stock
Market_Value = pd.read_excel(io="Data raw/Market Value EUR.xlsx", sheet_name="Market value EUR",
                             header=0, index_col="Date").dropna(axis=1) * 1000000
Market_Value = Market_Value.dropna()

# ICE-ECX CER Daily Futures settlement price
eua_futures_price = pd.read_excel(io="Data raw/CEDIF eua future price.xlsx", sheet_name="ICE ECX EUA",
                                  header=0, index_col="Date")["ICE-ECX CER Daily Futures"]

# STOXX EUROPE 600 E - TOTAL RETURN INDEX EUR
stoxx600_price = pd.read_excel(io="Data raw/STOXX EUROPE 600 E - TOT RETURN INDEX EUR.xlsx",
                               sheet_name="STOXX EU 600 E TRI", header=0,
                               index_col="Date")["STOXX EUROPE 600 E - TOT RETURN IND"]

# WISDOMTREE BRENT CRUDE OIL ETC (~E )
brent_oil_price = pd.read_excel(io="Data raw/Brent Crude Oil EUR.xlsx", sheet_name="Brent crude oil EUR",
                                header=0, index_col="Date")

print("preprocessing done!")
