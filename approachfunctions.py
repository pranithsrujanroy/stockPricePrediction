import math
import numpy as np
import pandas as pd

#pre process the stock data 
def pre_process(stock):
    # Convert Date column to datetime
    stock.loc[:, 'Date'] = pd.to_datetime(stock['Date'],format='%Y-%m-%d')

    # Change all column headings to be lower case, and remove spacing
    stock.columns = [str(x).lower().replace(' ', '_') for x in stock.columns]

    # Get month of each sample
    stock['month'] = stock['date'].dt.month

    # Sort by datetime
    stock.sort_values(by='date', inplace=True, ascending=True)
    return stock

#Moving average function
def moving_average(stock_data, train_size, window_size):
    #we find out moving average of the stock data over adj closing price 
    #the moving avg is calculated below as the avg of current element and last N-1 elements
    #Therefore we won't be needing the last value obtained from the below averages
    avg = stock_data['adj_close'].rolling(window = window_size, min_periods=1).mean()
    avg = avg[:-1]
    avg = np.array(avg)
    return avg[train_size - 1:]

def findRMSE(stock_values, predictions, cv_size):
    y_true = stock_values['adj_close']
    RMSE = math.sqrt(np.sum((y_true-predictions)**2)/cv_size)
    return RMSE

def findMAPE(stock_values, predictions, cv_size):
    y_true = stock_values['adj_close']
    MAPE = (np.sum(abs(y_true-predictions)/y_true)/cv_size)*100
    return MAPE

