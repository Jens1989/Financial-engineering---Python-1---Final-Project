# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


#Change the path below to be able to read the csv file
quotes = pd.read_csv(r"C:\Users\jens_\Dropbox\Jens\financial engineering\Python 1\Project\HistoricalQuotes.csv", index_col="date")

'''change the dates in the index from string to datetime objects
then the index needs to be reversed as the dates were descending'''

str_to_datetime = pd.to_datetime(quotes.index, dayfirst = True)
quotes.index = str_to_datetime
quotes = quotes.reindex(index=quotes.index[::-1])

def date_filter(date_list): #filter out the first available day for each month
    
    valid_dates = []
    current_month = 0    
    
    for date in date_list:
        
        if date.month != current_month and date.day <5:
            
            valid_dates.append(date)
            current_month = date.month
    
    return (valid_dates)

#filter the correct dates from the dataframe and select only the closing prices
filtered_dates = date_filter(quotes.index)
filtered_quotes = quotes.reindex(filtered_dates)
closes = filtered_quotes["close"] 

def exponential_smoothing(alpha, Yt, Ft): #Yt stockprice, Ft forecasted price
    
    Ft1 = alpha*Yt+(1-alpha)*Ft
    
    return (Ft1)

def run_smoothing(prices): #run the smoothing and linear regression model
    
    print ("please enter the amount of periods you want to run the model for,\
with a maximum of", len(prices)-1, "periods")
    
    periods = int(input("periods: "))
    
    prices = prices[-periods-1:-1]
    
    run = True
    
    while run == True:
    
        alpha = float(input("please enter the alpha value: "))
            
        forecast_prices = [prices[0]]
        
        for num, price in enumerate(prices):
            
                forecast_prices.append(exponential_smoothing(alpha,price,forecast_prices[num]))
        
        forecast_prices.pop(-1)
        
        x_axis = [x for x in range(1,len(prices)+1)]
        
        plt.plot(x_axis,prices, x_axis,forecast_prices)
        plt.title("historical versus smoothed prices")
        plt.legend(["historical price", "smoothed price"])
        plt.show()
        
        rerun = input("would you like to run the model again? Y/N ").lower()
        
        
        
        if rerun == "y":
            
            run = True
            
        else:
            
            run = False
          
    print (" ")
    print ("linear regression:")
    print (" ")
    print (least_squares(prices,forecast_prices))
    
def least_squares(explanatory_variable, response_variable):
    
    def correlation_coefficient(explanatory_variable, response_variable):
    
        x = sum(explanatory_variable)
        y = sum(response_variable)
        
        xy = 0
        
        for i,j in zip(explanatory_variable,response_variable):
            
            xy += i*j
            
        x2 = sum([i**2 for i in explanatory_variable])
        y2 = sum([i**2 for i in response_variable])
        
        n = len(explanatory_variable)
        
        r = (xy-(x*y)/n)/(np.sqrt(x2-x**2/n)*np.sqrt(y2-y**2/n))
        
        return (r)
    
    r = correlation_coefficient(explanatory_variable, response_variable)
    
    b1 = r * np.std(response_variable)/np.std(explanatory_variable) # slope
        
    b0 = round(np.mean(response_variable)-b1*np.mean(explanatory_variable),2) #intercept
    
    
    print ("y_pred = ", '%.1f' % round(b1,2), "* x", "+", '%.1f' % round(b0,2) )
    print ("correlation coefficient: ", correlation_coefficient(explanatory_variable,response_variable))
    print ("prediction for the next period","(",len(explanatory_variable)+1,") : ", b1*explanatory_variable[-1]+b0)
    
    scatter(explanatory_variable,response_variable)
    

def scatter(X,y):
    
    X = np.array(X)
    X = X.reshape(-1,1)
    
    y = np.array(y)
    y = y.reshape(-1,1)
    
    lm = linear_model.LinearRegression()
    lm.fit(X,y)
    
    Xfit = lm.predict(X)
    
    plt.scatter(X,y,color="black")
    plt.plot(X,Xfit,color="blue",linewidth=3)
    
run_smoothing(closes) #run the model with the closes of the stockprices
