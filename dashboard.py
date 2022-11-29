# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:25:15 2022

@author: ythonukunuru
"""
###############################################################################
# FINANCIAL DASHBOARD 
###############################################################################

#==============================================================================
# Initiating
#==============================================================================

import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import yfinance as yf
import streamlit as st

#==============================================================================
# Initial Setup
#==============================================================================

# Configuration of page
stock_image = Image.open("stock_emoji.png")
st.set_page_config(page_title='Financial Dashboard',
                   page_icon=stock_image,
                   layout="centered", 
                   initial_sidebar_state="auto")

# Add dashboard title and description
st.title("Financial Dashboard")
st.write("Data source: Yahoo Finance. URL: https://finance.yahoo.com/")

# --- Multiple choices box ---
    
# Get the list of stock tickers from S&P500
ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    
    
# Add multiple choices box
tickers = st.multiselect("Select Stock Name", ticker_list)


# --- Add a button ---
get = st.button("Get data", key="get")


# Mention names of tabs
Summary,Chart,Financials,MonteCarloSimulation,Analysis,Profile,Holders,News = st.tabs(["Summary",
                                                         "Chart",
                                                         "Financials",
                                                         "Monte Carlo Simulation",
                                                         "Analysis",
                                                         "Profile",
                                                         "Holders",
                                                         "News"])

#==============================================================================
# Summary Tab
#==============================================================================

with Summary:
            
    # Adding columns to show data in two tables 
    col1, col2= st.columns(2)
    
    
    # 
    with col1:
        
        
        # Defining a fucntion to get get stocks data
        @st.cache(suppress_st_warning=True)
        def Summary(ticker):
            global var
            for ticker in tickers:
                stock_name = yf.Ticker(ticker)
                tab1_dict = {
                              "Previous Close": stock_name.info['previousClose'],
                              "Open":           stock_name.info['open'],
                              "Bid":            stock_name.info['bid'],
                              "Ask":            stock_name.info['ask'],
                              "Volume":         stock_name.info['volume'],
                              "Avg. Volume":    stock_name.info['averageVolume'],
                              "Day Low":        stock_name.info['dayLow'],
                              "Day High":       stock_name.info['dayHigh']
                            }
                return tab1_dict
                
        # Conditions to show financial dcouemnts in web app
        if tickers != '-':
            table1 = Summary(tickers)
            table1_df = pd.DataFrame.from_dict(table1,orient='index',columns=['Values'])
            st.table(table1_df) 
            
            
    with col2:
        
        @st.cache(suppress_st_warning=True)
        def Summary1(ticker):
            global var
            for ticker in tickers:
                stock_name = yf.Ticker(ticker)
                tab2_dict = {
                              "Market Cap":         stock_name.info['marketCap'],
                              "Beta (5Y Monthly)":  stock_name.info['beta'],
                              "PE Ratio (TTM)":     stock_name.info['trailingPE'],
                              "EPS (TTM)":          stock_name.info['trailingEps'],
                              "Volume":             stock_name.info['dividendRate'],
                              "Forward Dividend":   stock_name.info['averageVolume'],
                              "52 Week Low":        stock_name.info['fiftyTwoWeekLow'],
                              "52 Week High":       stock_name.info['fiftyTwoWeekHigh']
                            }
                return tab2_dict
                
        # Conditions to show financial dcouemnts in web app
        if tickers != '-':
            table2 = Summary1(tickers)
            table2_df = pd.DataFrame.from_dict(table2,orient='index',columns=['Values'])
            st.table(table2_df) 
        
 
        
    # Fetching OHLC for all stocks for maximum duration 
    
    @st.cache
    def OHLC(ticker):
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
            HistoricalPrices = stock_name.history(period='MAX',interval='1d')   
            return HistoricalPrices
        
    # Plotting chart for above retrieved historical data 
    if tickers != '-':
            OHLC_Max = OHLC(tickers) 
            
           # Defining axises in chart
            fig = px.area(OHLC_Max, OHLC_Max.index, OHLC_Max['Close'])
            
            
            # Updating figure axis
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, 
                             label="1M", 
                             step="month", 
                             stepmode="backward"),
                        dict(count=3, 
                             label="3M", 
                              step="month", 
                             stepmode="backward"),
                        dict(count=6, 
                             label="6M", 
                             step="month", 
                             stepmode="backward"),
                        dict(count=1, 
                             label="YTD", 
                             step="year", 
                             stepmode="todate"),
                        dict(count=1, 
                             label="1Y", 
                             step="year", 
                             stepmode="backward"),
                        dict(count=3, 
                             label="3Y", 
                             step="year", 
                             stepmode="backward"),
                        dict(count=5, 
                             label="5Y", 
                             step="year", 
                             stepmode="backward"),
                        dict(label = "MAX", step="all")
                    ])
                )
            )
            
            
            
            # Showing chart in dashboard
            st.plotly_chart(fig)
            
#==============================================================================
# Chart Tab
#==============================================================================            

                
with Chart:
    
    # Creating columns
    col1,col2,col3,col4,col5 = st.columns(5)
    
    
    # Defining columns 
    with col1:
        start_date = col1.date_input("Start date", datetime.today().date() - timedelta(days=30))
        
    with col2:
        end_date = col2.date_input("End date", datetime.today().date())
        
    with col3:
        time_period = st.selectbox("Time Period", ('1Mo', '3Mo', '6Mo', 
                                                   'YTD','1Y', '3Y','5Y', 'MAX'))
        
    with col4:
        time_intervals = st.selectbox('Interval', ('1d','1Mo','1Y'))
        
    with col5:
        chart = st.selectbox('Plot Type', ('Line','Candle'))
        
        
    # Funntion to gather historical prices 
    def GetHistPrices(ticker):
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
            HistPrices = stock_name.history(period= time_period, interval = time_intervals)   
            return HistPrices
        
    # Preparing data to calculate moving avergae for 50 days 
    for tp in time_period:
        if tickers != '-':
            plot_data = GetHistPrices(tickers)
            plot_data['MA50'] = plot_data['Close'].rolling(50).mean()
            plot_data = plot_data.reset_index()
            plot_data["Date"] = pd.to_datetime(plot_data["Date"]).dt.strftime('%Y-%m-%d')
            
            # Plotting data on chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            if chart == 'Line':
                fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Close'], mode='lines', 
                                         name = 'Close'), secondary_y = False)
            else:
                fig.add_trace(go.Candlestick(x = plot_data['Date'], open = plot_data['Open'], 
                                             high = plot_data['High'], low = plot_data['Low'], close = plot_data['Close'], name = 'Candle'))
              
                    
            fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['MA50'], mode='lines', name = '50 days MA'), secondary_y = False)
            
            
            # Adding colors to volumes
            colors = ['green' if row['Open'] - row['Close'] >= 0 
                              else 'red' for index, row in plot_data.iterrows()]
            
            fig.add_trace(go.Bar(x = plot_data['Date'], y = plot_data['Volume'], name = 'Volume', marker_color=colors), secondary_y = True)
    
            fig.update_yaxes(range=[0, plot_data['Volume'].max()*3], showticklabels=False, secondary_y=True)
            
            # removing rangeslider
            fig.update_layout(xaxis_rangeslider_visible=False)
        
            # hide weekends
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])    
        
            st.plotly_chart(fig)


#==============================================================================
# Financials
#==============================================================================    

with Financials:
    
    fin_docs = st.selectbox("Financial Document",('Income Statement','Balance Sheet','Cash Flow'))
    period = st.selectbox("Time Perios", ['Yearly', 'Quarterly'])
    
    
    # Defining a fucntion to fetch income statements year wise 
    @st.cache
    def getincomestatementsyearly(tickers):
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
        return stock_name.get_financials(freq='yearly')
    
    # Defining a fucntion to gather quaterly income statements 
    @st.cache
    def getincomestatementsquaterly(tickers):
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
        return stock_name.get_financials(freq='quarterly')
    
    # Defining a fucntion to fetch balance sheets year wise 
    @st.cache
    def getbalancesheetsyearly(tickers):
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
        return stock_name.balancesheet
    
    # Defining a fucntion to gather quaterly income statements 
    @st.cache
    def getbalancesheetsquaterly(tickers):
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
        return stock_name.quarterly_balancesheet
    
    # Defining a fucntion to fetch balance sheets year wise 
    @st.cache
    def getcashflowyearly(tickers):
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
        return stock_name.cashflow
    
    # Defining a fucntion to gather quaterly income statements 
    @st.cache
    def getcashflowquarterly(tickers):
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
        return stock_name.quarterly_cashflow
    
    
    # Conditions to show financial dcouemnts in web app
    if tickers != '-' and fin_docs == 'Income Statement' and period == 'Yearly':
              stmt = getincomestatementsyearly(tickers)
              st.table(stmt)
              
              
    if tickers != '-' and fin_docs == 'Income Statement' and period == 'Quarterly':
              stmt = getincomestatementsquaterly(tickers)
              st.table(stmt)
              
              
    if tickers != '-' and fin_docs == 'Balance Sheet' and period == 'Yearly':
              bal_sheets = getbalancesheetsyearly(tickers)
              st.table(bal_sheets)
              
              
    if tickers != '-' and fin_docs == 'Balance Sheet' and period == 'Quarterly':
              bal_sheets = getbalancesheetsquaterly(tickers)
              st.table(bal_sheets)
              
              
    if tickers != '-' and fin_docs == 'Cash Flow' and period == 'Yearly':
              cashflow = getcashflowyearly(tickers)
              st.table(cashflow)
              
              
    if tickers != '-' and fin_docs == 'Cash Flow' and period == 'Quarterly':
              cashflow = getcashflowquarterly(tickers)
              st.table(cashflow)
    
    
 #==============================================================================
 # Monte Carlo Simulation
 #==============================================================================

with MonteCarloSimulation:
    
    # Number of simulations and time period to make simulation
    n_simulations = st.selectbox("Number of Simulations", [200, 500, 1000])
    time_period = st.selectbox("Time Horizon", [30, 60, 90])
    
    # Defining a fucntion to find future stock prices using Monte Carlo
    # Simulation 
    @st.cache
    def montecarlo(tickers, n_simulations,time_period):
        
        # Prediction will made on the basis of last 30 days closed price.
        # Fetching last 30 days OHLC data and saving close price
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
            stock_price = stock_name.history(period='1mo')
            close_price = stock_price['Close']
            
            # Calculating daily volatility of stock
            daily_return = close_price.pct_change()
            daily_volatility = np.std(daily_return)
            
            # Run the simulation
            simulation_df = pd.DataFrame()
            
            for i in range(n_simulations):        
                         
                   # The list to store the next stock price
                   next_price = []
       
           # Create the next stock price
                   last_price = close_price[-1]
       
                   for x in range(time_period):
                                  
                         # Generate the random percentage change around  mean (0) 
                         # and std (daily_volatility)
                         future_return = np.random.normal(0, daily_volatility)

                         # Generating  random future price
                         future_price = last_price * (1 + future_return)

                         # Save the price and go next
                         next_price.append(future_price)
                         last_price = future_price
       
                   # Store the result of the simulation
                   simulation_df[i] = next_price
                         
            return simulation_df
            

# Calculating Value at Risk (VaR), plotting it and Monte Carlo Simulation

    if tickers != '-':
        mc_sim = montecarlo(tickers, n_simulations, time_period)
 
        # Fetching last 30 days OHLC data and saving close price
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
            stock_price = stock_name.history(period='1mo')
            close_price = stock_price['Close']
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        ax.plot(mc_sim)
        plt.title('Monte Carlo simulation for stock' + ' in next ' + str(time_period) + ' days')
        plt.xlabel('Day')
        plt.ylabel('Price')
        
        
        plt.axhline(y= close_price[-1], color ='red')
        plt.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
        ax.get_legend().legendHandles[0].set_color('red')

        st.pyplot(fig)
        
        # Value at Risk
        st.subheader('Value at Risk (VaR)')
        ending_price = mc_sim.iloc[-1:, :].values[0, ]
        fig1, ax = plt.subplots(figsize=(15, 10))
        ax.hist(ending_price, bins=50)
        plt.axvline(np.percentile(ending_price, 5), color='red', linestyle='--', linewidth=1)
        plt.legend(['5th Percentile of the Future Price: ' + str(np.round(np.percentile(ending_price, 5), 2))])
        plt.title('Distribution of the Ending Price')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        st.pyplot(fig1)
        
        
        future_price_95ci = np.percentile(ending_price, 5)
        # Value at Risk
        VaR = close_price[-1] - future_price_95ci
        st.write('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')

    
#==============================================================================
# Analysis
#============================================================================== 

with Analysis:
    
    # Defining a function to get stocks analysis information 
    @st.cache
    def GetAnalysis(tickers):
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
        return stock_name.get_analysis()
            
    if tickers != '-' :
        
        # Displaying Earnings Estimate in dashboard
        st.markdown('**Earnings Estimate**')
        earnings_estimate = GetAnalysis(tickers)
        earnings_estimate = earnings_estimate.loc[:, [ 'Earnings Estimate Avg',
                                                     'Earnings Estimate Low',
                                                     'Earnings Estimate High',
                                                     'Earnings Estimate Year Ago Eps',
                                                     'Earnings Estimate Number Of Analysts']]
        earnings_estimate = earnings_estimate.drop(['+5Y', '-5Y'], axis=0)
        # Renaming Rows
        earnings_estimate.index = ['Current Quarter','Next Quarter', 'Current Year', 'Next Year']
        st.dataframe(earnings_estimate)
        
        
        # Displaying Revenue Estimate in dashboard
        st.markdown('**Revenue Estimate**')
        revenue_estimate = GetAnalysis(tickers)
        revenue_estimate = revenue_estimate.loc[:, ['Revenue Estimate Avg',
                                                     'Revenue Estimate Low',
                                                     'Revenue Estimate High',
                                                     'Revenue Estimate Number Of Analysts',
                                                     'Revenue Estimate Year Ago Revenue',
                                                     'Revenue Estimate Growth']]
        revenue_estimate = revenue_estimate.drop(['+5Y', '-5Y'], axis=0)
        # Renaming Rows
        revenue_estimate.index = ['Current Quarter','Next Quarter', 'Current Year', 'Next Year']
        st.dataframe(revenue_estimate)
        
        
        # Displaying EPS Trend in dashboard
        st.markdown('**EPS Trend**')
        eps_trend = GetAnalysis(tickers)
        eps_trend = eps_trend.loc[:, ['Eps Trend Current',
                                     'Eps Trend 7Days Ago',
                                     'Eps Trend 30Days Ago',
                                     'Eps Trend 60Days Ago',
                                     'Eps Trend 90Days Ago']]
        eps_trend = eps_trend.drop(['+5Y', '-5Y'], axis=0)
        # Renaming Rows
        eps_trend.index = ['Current Quarter','Next Quarter', 'Current Year', 'Next Year']
        st.dataframe(eps_trend)
        
        
        # Displaying EPS Revisions in dashboard
        st.markdown('**EPS Revisions**')
        eps_revisions = GetAnalysis(tickers)
        eps_revisions = eps_revisions.loc[:, [ 'Eps Revisions Up Last7Days',
                                         'Eps Revisions Up Last30Days',
                                         'Eps Revisions Down Last30Days',
                                         'Eps Revisions Down Last90Days']]
        eps_revisions = eps_revisions.drop(['+5Y', '-5Y'], axis=0)
        # Renaming Rows
        eps_revisions.index = ['Current Quarter','Next Quarter', 'Current Year', 'Next Year']
        st.dataframe(eps_revisions)

    
#==============================================================================
# Company Profile
#==============================================================================  

with Profile:
    
    @st.cache
    def CompanyInfo(ticker):
        for ticker in tickers:
            return yf.Ticker(ticker).info
    
    if tickers != '':
        # Get the company information in list format
        info = CompanyInfo(tickers)
        
        
        # Show the company description
        st.markdown(ticker)
        st.caption(info['industry'])
        st.image(info['logo_url'])
        st.write(info['longBusinessSummary'])
        
                     
#==============================================================================
# Holders
#==============================================================================

with Holders:
    
    
    # Defining a fucntion to get details of major holders
    @st.cache
    def GetMajorHolders(tickers):
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
        return stock_name.major_holders


    # Displaying major holders information in dashboard            
    if tickers != '-' :
        
        # Displaying Earnings Estimate in dashboard
        st.markdown('**Major Holders**')
        majorholders = GetMajorHolders(tickers)
        majorholders = majorholders.rename_axis(None, axis=1)
        
        st.table(majorholders)
        
        
    # Function to retrieve institutional holders information 
    @st.cache
    def GetInstitutionalHoldersInfo(ticker):
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
            return stock_name.institutional_holders
    
    
    # Using above fucntion to show details of top institutional holders in 
    # dashbaord
    if tickers != '':
        
        # Show major share holders
        st.markdown('**Top Institutional Holders**')
        inst_holders = GetInstitutionalHoldersInfo(tickers)
        st.table(inst_holders)
        
        
    # Function to retrieve institutional holders information 
    @st.cache
    def GetMutualFundsHoldersInfo(ticker):
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
            return stock_name.mutualfund_holders
    
    
    # Using above fucntion to display details of top mutual funds  holders in 
    # dashbaord
    if tickers != '':
        
        # Show major share holders
        st.markdown('**Top Mutual Funds Holders**')
        mf_holders = GetMutualFundsHoldersInfo(tickers)
        st.table(mf_holders)

        
#==============================================================================
# News
#==============================================================================        

with News:

    # Defining a fucntion to fetch news articles 
    def getnews(ticker):
        for ticker in tickers:
            stock_name = yf.Ticker(ticker)
            return stock_name.news
            
        
    if tickers != '-':
        news1 = getnews(tickers)       
        for i in news1:
            st.write(f'{i["title"]}\n{i["link"]}')
                                                       

#==============================================================================
# Main body
#==============================================================================

def run():
    
    # Show selected tab
    if Summary == 'Summary':
        Summary()
    elif Chart == 'Chart':
        Chart()
    elif Financials == 'Financilas':
        Financials()
    elif MonteCarloSimulation == 'Monte Carlo Simulation':
        MonteCarloSimulation()
    elif Analysis == 'Analysis':
        Analysis()
    elif Profile == 'Profile':
        Profile()
    elif Holders == 'Holders':
        Holders()
    elif News == 'News':
        News()
       
        
if __name__ == "__main__":
    run()   



