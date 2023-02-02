import yfinance as yf
import pandas as pd
import numpy as np
import timeit

data = pd.read_excel("Data/Ticker_all.xlsx", sheet_name="Equity")
ticker = pd.read_excel("Data/Ticker_industry.xlsx", sheet_name="Sheet1")
ticker_end = len(ticker)

# Ticker = data["Ticker"]


def load_info(DataFrame, loop_start=0, step=500):
    columns = ["currency", "totalRevenue", "beta", "operatingCashflow", "industry", "fullTimeEmployees"]
    output = pd.DataFrame(columns=columns)
    # len(data)
    for i in range(loop_start, loop_start + step):
        temp = yf.Ticker(DataFrame["Ticker"][i])
        # hist = temp.history(period="max")

        symbol = currency = totalRevenue = beta = industry = operatingCashflow = fullTimeEmployees = None

        info = temp.info
        # symbol
        try:
            symbol = info["symbol"]
        except KeyError:
            pass
        # currency
        try:
            currency = info["currency"]
        except KeyError:
            pass
        # totalRevenue
        try:
            totalRevenue = info["totalRevenue"]
        except KeyError:
            pass
        # beta
        try:
            beta = info["beta"]
        except KeyError:
            pass
        # industry
        try:
            industry = info["industry"]
        except KeyError:
            pass
        # operatingCashflow
        try:
            operatingCashflow = info["operatingCashflow"]
        except KeyError:
            pass
        # fullTimeEmployees
        try:
            fullTimeEmployees = info["fullTimeEmployees"]
        except KeyError:
            pass

        Firm_Info = pd.DataFrame([currency, totalRevenue, beta, operatingCashflow, industry, fullTimeEmployees],
                                 index=columns, columns=[symbol])
        output = pd.concat([output, Firm_Info.transpose()], axis=0, join="inner")

    return output


def save(DataFrame, start=0, header=False):
    # df.to_excel("Data/Ticker_industry.xlsx", sheet_name="data", startrow=start, header=False)
    with pd.ExcelWriter('Data/Ticker_industry.xlsx', mode='a', if_sheet_exists="overlay") as writer:
        DataFrame.to_excel(writer, sheet_name="Sheet1", startrow=start, header=header)


def main(data, ticker):

    ticker_end = len(ticker)
    for i in range(0, int(np.round(len(data)/10))):
        start = timeit.default_timer()
        if ticker_end == 0:
            df = load_info(data)
            save(df, header=True)
        else:
            ticker = pd.read_excel("Data/Ticker_industry.xlsx", sheet_name="Sheet1")
            ticker_end = len(ticker)
            last_saved_ticker = ticker.iloc[-1, 0]
            # position of the last ticket saved in the ticker DataFrame
            position_last_ticker_saved = data[data.eq(last_saved_ticker).any(1)].index[0]

            df = load_info(data, loop_start=position_last_ticker_saved + 1)
            save(df, ticker_end + 1)
        stop = timeit.default_timer()
        Time = stop - start
        print('Time: ', Time)

        print(df)
        try:
            print(position_last_ticker_saved)
        finally:
            pass


main(data, ticker)





