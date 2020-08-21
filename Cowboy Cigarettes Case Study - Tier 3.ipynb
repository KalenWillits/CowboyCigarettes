# %% markdown
# # Springboard Time Series - 'Cowboy Cigarettes' Case Study - Tier 3
# %% markdown
# ## Brief
#
# You're working in the US federal government as a data scientist in the Health and Environment department. You've been tasked with determining whether sales for the oldest and most powerful producers of cigarettes in the country are increasing or declining.
#
# **Cowboy Cigarettes (TM, *est.* 1890)** is the US's longest-running cigarette manufacturer. Like many cigarette companies, however, they haven't always been that public about their sales and marketing data. The available post-war historical data runs for only 11 years after they resumed production in 1949; stopping in 1960 before resuming again in 1970. Your job is to use the 1949-1960 data to predict whether the manufacturer's cigarette sales actually increased, decreased, or stayed the same. You need to make a probable reconstruction of the sales record of the manufacturer - predicting the future, from the perspective of the past - to contribute to a full report on US public health in relation to major cigarette companies.
#
# The results of your analysis will be used as part of a major report relating public health and local economics, and will be combined with other studies executed by your colleagues to provide important government advice.
#
# -------------------------------
# As ever, this notebook is **tiered**, meaning you can elect that tier that is right for your confidence and skill level. There are 3 tiers, with tier 1 being the easiest and tier 3 being the hardest.
#
# **1. Sourcing and loading**
# - Load relevant libraries
# - Load the data
# - Explore the data
#
#
# **2. Cleaning, transforming and visualizing**
# - Dropping unwanted columns
# - Nomenclature
# - Type conversions
# - Making a predictor variable `y`
# - Getting summary statistics for `y`
# - Plotting `y`
#
#
# **3. Modelling**
# - Decomposition
#     - Trend
#     - Seasonality
#     - Noise
# - Testing for stationarity with KPSS
# - Making the data stationary
# - The ARIMA Model
#     - Make a function to find the MSE of a single ARIMA model
#     - Make a function to evaluate the different ARIMA models with different p, d, and q values
# - Visualize the results
# - Application: Forecasting
#
# **4. Evaluating and concluding**
# - What is our conclusion?
# - Next steps
#
# %% markdown
# ## 0. Preliminaries
#
# Time series data is just any data displaying how a single variable changes over time. It comes as a collection of metrics typically taken at regular intervals. Common examples of time series data include weekly sales data and daily stock prices. You can also easily acquire time series data from [Google Trends](https://trends.google.com/trends/?geo=US), which shows you how popular certain search terms are, measured in number of Google searches.
# %% markdown
# ## 1. Sourcing and Loading
#
# ### 1a. Load relevant libraries
# %% codecell
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% markdown
# ### 1b. Load the data
# Call the variable `cigData`.
# %% codecell
cd_data = 'data/'
cd_figures = 'figures/'
cigData = pd.read_csv(cd_data+'CowboyCigsData.csv')
# %% markdown
# ### 1c. Explore the data
# We now need to check whether the data conduces to a time series style analysis.
# %% codecell
cigData.head()
# %% markdown
# Over a million cigarettes sold in the month of January 1949. This certainly is a popular cigarette brand.
# %% markdown
# Check out the columns feature of the data. How many columns are there?
# %% codecell
cigData.columns
print('There are ', len(cigData.columns), ' columns.')
# %% markdown
# Let's check out the data types of our columns.
# %% codecell
cigData.dtypes
# %% markdown
# Check whether there are any null values.
# %% codecell
cigData.isnull().sum()
# %% markdown
# ## 2. Cleaning, transforming and visualizing
# %% markdown
# ### 2a. Dropping unwanted columns
# We need to cut that `Unnamed: 0` column. Delete it here.
# %% codecell
cigData.drop('Unnamed: 0', axis=1, inplace=True)
# %% markdown
# ### 2b. Nomenclature
# %% markdown
# We can see that the `Time` column actually has the granularity of months. Change the name of that column to `Month`.
# %% codecell
cigData = cigData.rename(columns={'Time':'Month'})
# %% markdown
# Call a head() to check this has worked.
# %% codecell
cigData.head()
# %% markdown
# ### 2c. Type conversions
# %% markdown
# Now, doing a time series analysis on a Pandas dataframe is overkill, and is actually counter-productive. It's much more easy to carry out this type of analysis if we convert our data to a series first.
#
# Notice that the `Month` field was an object. Let's type convert the `Month` column to a Python `datetime`, before making that the index.
# %% codecell
cigData['Month'] = pd.to_datetime(cigData['Month'])
df = cigData.set_index(cigData['Month'])
# %% markdown
# Perfect!
# %% markdown
# ### 2d. Making a predictor variable `y`
# %% markdown
# The data is now indexed by date, as time series data ought to be.
#
# Since we want to predict the number of cigarette sales at Cowboy cigarettes, and `y` is typically used to signify a predictor variable, let's create a new variable called `y` and assign the indexed #Passenger column.
# %% codecell
y = df['#CigSales']
# %% markdown
# Check the type of our new variable.
# %% codecell
type(y)
# %% markdown
# ### 2e. Getting summary statistics for `y`
# %% markdown
# Get the summary statistics of our data here.
# %% codecell
y.describe()
# %% markdown
# Try visualizing the data. A simple `matplotlib` plot should do the trick.
# %% markdown
# ### 2f. Plotting `y`
# %% codecell
# plt.figure(figsize=(15,5))
plt.plot(cigData['#CigSales'], color='black')
title = 'trend_of_cig_sales'
plt.title(title.replace('_',' ').title())
plt.ylabel('Cigs Sold')
plt.xlabel('Month')
plt.xticks(list(range(0, len(y)+1, 6)))
plt.grid(axis='x')
plt.savefig(cd_figures+title+'.png', transparent=True)
# %% markdown
# # Observations
# The "Trend Of Cig Sales" plot shows a positive linear correlation over time. There also seems to be a volatile pattern repeating with a sharp spike and back down again over a course of a few months at a time. These spikes appear to be consistent in pattern around seasons. Every 6 months there appears to be a spike and plateau.
# %% markdown
# ## 3. Modelling
# ### 3a. Decomposition
# What do you notice from the plot? Take at least `2` minutes to examine the plot, and write down everything you observe.
#
# All done?
#
# We can see that, generally, there is a trend upwards in cigarette sales from at Cowboy Cigarettes. But there are also some striking - and perhaps unexpected - seasonal fluctuations. These seasonal fluctations come in a repeated pattern. Work out when these seasonal fluctuations are happening, and take 2 minutes to hypothesize on their cause here.
#
# What does it mean to *decompose* time series data? It means breaking that data into 3 components:
#
# 1. **Trend**: The overall direction that the data is travelling in (like upwards or downwards)
# 2. **Seasonality**: Cyclical patterns in the data
# 3. **Noise**: The random variation in the data
#
# We can treat these components differently, depending on the question and what's appropriate in the context. They can either be added together in an *additive* model, or multiplied together in a *multiplicative* model.
#
# Make a coffee, take `5` minutes and read [this article](https://medium.com/@sigmundojr/seasonality-in-python-additive-or-multiplicative-model-d4b9cf1f48a7) and think about whether our data would conduce to an additive or multiplicative model here. Write your conclusion down just here:
#
# -------------------------------
# %% markdown
# All done? Well, just on the basis of the plot above, it seems our Cowboy Cigarettes data is actually multiplicative.
#
# That's because, as time progresses, the general trend seems to be increasing *at a rate that's also increasing*. We also see that the seasonal fluctuations (the peaks and troughs) get bigger and bigger as time progresses.
#
# Now on the other hand, if the data were simply additive, we could expect the general trend to increase at a *steadily*, and a constant speed; and also for seasonal ups and downs not to increase or decrease in extent over time.
#
# Happily, we can use the `decompose()` function to quantify the component parts described above in our data.
# %% codecell
# Import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose

# Make a variable called decomposition, and assign it y passed to seasonal_decompose()
decomposition = seasonal_decompose(y, model='multiplicative')

# Make three variables for trend, seasonal and residual components respectively.
# Assign them the relevant features of decomposition
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
# Plot the original data, the trend, the seasonality, and the residuals
title = 'trend_decompose'
plt.title(title)
plt.subplot(411)
plt.plot(y, label = 'y')
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(trend, label = 'Trend')
plt.legend(loc = 'best')
plt.subplot(413)
plt.plot(seasonal, label = 'Seasonal')
plt.legend(loc = 'best')
plt.subplot(414)
plt.plot(residual, label = 'Residuals')
plt.legend(loc = 'best')
plt.savefig(cd_figures+title.replace('_', ' ').title()+'.png')
# %% markdown
# ### 3b. Testing for stationarity with KPSS
# As you know, when doing time series analysis we always have to check for stationarity. Imprecisely, a time series dataset is stationary just if its statistical features don't change over time. A little more precisely, a stationary time series dataset will have constant mean, variance, and covariance.
#
# There are many ways to test for stationarity, but one of the most common is the KPSS test. The Null hypothesis of this test is that the time series data in question is stationary; hence, if the *p*-value is less than the significance level (typically 0.05, but we decide) then we reject the Null and infer that the data is not stationary.
# %% codecell
from statsmodels.tsa.stattools import kpss
kpss(y)
# %% markdown
# Since our p-value is less than 0.05, we should reject the Null hypothesis and deduce the non-stationarity of our data.
#
# But our data need to be stationary! So we need to do some transforming.
# %% markdown
# ### 3c. Making the data stationary
# Let's recall what it looks like.
# %% codecell
y.head()
# %% markdown
# In our plot, we can see that both the mean and the variance *increase as time progresses*. At the moment, our data has neither a constant mean, nor a constant variance (the covariance, however, seems constant).
#
# One often used way of getting rid of changing variance is to take the natural log of all the values in our dataset. Let's do this now.
# %% codecell
y_log = np.log(y)
# %% markdown
#
# When you plot this, you can see how the variance in our data now remains constant over time.
# %% codecell
# plt.figure(figsize=(6.5,5))
plt.plot(y_log, color='black')
title = 'log'
plt.title(title.replace('_',' ').title())
plt.ylabel('Cigs Sold')
plt.xlabel('Month')
plt.savefig(cd_figures+title+'.png', transparent=True)
# %% markdown
# We now have a constant variance, but we also need a constant mean.
#
# We can do this by *differencing* our data. We difference a time series dataset when we create a new time series comprising the difference between the values of our existing dataset.
#
# Python is powerful, and we can use the `diff()` function to do this. You'll notice there's one less value than our existing dataset (since we're taking the difference between the existing values).
# %% codecell
y_log.diff()
# %% markdown
# Our p-value is now greater than 0.05, so we can accept the null hypothesis that our data is stationary.
# %% markdown
# ### 3d. The ARIMA model
#
# Recall that ARIMA models are based around the idea that it's possible to predict the next value in a time series by using information about the most recent data points. It also assumes there will be some randomness in our data that can't ever be predicted.
#
# We can find some good parameters for our model using the `sklearn` and `statsmodels` libraries, and in particular `mean_squared_error` and `ARIMA`.
# %% codecell
# Import mean_squared_error and ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
# %% markdown
# #### 3di. Make a function to find the MSE of a single ARIMA model
# Things get intricate here. Don't worry if you can't do this yourself and need to drop down a Tier.
# %% codecell
def find_mse(data, arima_order):
    # Needs to be an integer because it is later used as an index.
    split = int(len(data) * 0.8)
    # Make train and test variables, with 'train, test'
    train, test = data[0:split], data[split:len(data)]
    past=[x for x in train]
    # make predictions. Declare a variable with that name
    predictions = list()
    for i in range(len(test)):#timestep-wise comparison between test data and one-step prediction ARIMA model.
        model = ARIMA(past, order=arima_order)
        model_fit = model.fit(disp=0)
        future = model_fit.forecast()[0]
        predictions.append(future)
        past.append(test[i])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    # Return the error
    return error
# %% markdown
# #### 3dii. Make a function to evaluate the different ARIMA models with different p, d, and q values
# %% codecell
# Make a function to evaluate different ARIMA models with several different p, d, and q values.
def evaluate_models(dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    # Iterate through p_values
    for p in p_values:
        # Iterate through d_values
        for d in d_values:
            # Iterate through q_values
            for q in q_values:
                # p, d, q iterator variables in that order
                order = (p,d,q)
                try:
                    # Make a variable called mse for the Mean squared error
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    return print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
# %% codecell
# Now, we choose a couple of values to try for each parameter.
p_values = [x for x in range(0, 3)]
d_values = [x for x in range(0, 3)]
q_values = [x for x in range(0, 3)]
# %% codecell
# Finally, we can find the optimum ARIMA model for our data.
# Nb. this can take a while...!
import warnings
warnings.filterwarnings("ignore")
evaluate_models(y_log, p_values, d_values, q_values)
# %% markdown
# So the best p,d, q, parameters for our ARIMA model are 2, 1, 1 respectively. Now we know this, we can build the model.
# %% codecell
p=2
d=1
q=1
model = ARIMA(y_log, order=(p,d,q))
model_fit = model.fit()
forecast = model_fit.forecast(24)
# %% markdown
# We can take a look at a summary of the model this library has built around our data.
# %% codecell
model_fit.summary()
# %% markdown
# ### 3e. Visualize the results
#
# Visualize the original dataset plotted against our model.
# %% codecell
title = 'data_vs_prediction'
plt.figure(figsize=(15,10))
plt.plot(y_log.diff(), color='black', label='data')
plt.plot(model_fit.predict(), color='blue', label='prediction')
plt.title(title.replace('_', ' ').title())
plt.savefig(cd_figures+title+'.png', transparent=True)
# %% markdown
# ### 3f. Application: Forecasting
#
# We've done well: our model fits pretty closely to our existing data. Let's now use it to forecast what's likely to occur in future.
# %% codecell
# Declare a variable called forecast_period with the amount of months to forecast, and
# create a range of future dates that is the length of the periods you've chosen to forecast
forecast_period = len(cigData['Month'])+12
date_range = pd.date_range(y_log.index[-1], periods = forecast_period,
              freq='MS').strftime("%Y-%m-%d").tolist()

# Convert that range into a dataframe that includes your predictions
future_months = pd.DataFrame(date_range, columns = ['Month'])

future_months['Month'] = pd.to_datetime(future_months['Month'])
future_months.set_index('Month', inplace = True)
future_months['Prediction'] = pd.Series(forecast[0])
# Error : Length of values does not match the index
# future_months['Prediction'] = np.NaN
# future_months['Prediction'] = pd.concat([future_months['Prediction'], pd.Series(forecast[0])])
# future_months['Prediction'].fillna(np.mean(forecast[0]), inplace=True)

future_months
# Plot your future predictions
title = 'future_predictions_data'
plt.figure(figsize=(15,10))
plt.plot(y_log, color='black')
plt.plot(y_log[[pd.to_datetime('1960-11-01')]].append(future_months['Prediction']))
plt.title(title.replace('_', ' ').title())
plt.savefig(cd_figures+title+'.png')

# %% codecell
title = 'future_predictions_forecast'
plt.figure(figsize=(15,10))
plt.plot(forecast[0])
plt.xlabel('Months after November 1960')
plt.title(title.replace('_', ' ').title())
plt.savefig(cd_figures+title+'.png')
# %% markdown

# ## 4. Evaluating and Concluding
#
# Our model captures the center of a line that's increasing at a remarkable rate. Cowboy Cigarettes sell more cigarettes in the summer, perhaps due to the good weather, disposable income and time off that people enjoy, and the least in the winter, when people might be spending less and enjoying less free time outdoors.
#
# Remarkably, our ARIMA model made predictions using just one variable. We can only speculate, however, on the causes of the behaviour predicted by our model. We should also take heed that spikes in data, due to sudden unusual circumstances like wars, are not handled well by ARIMA; and the outbreak of the Vietnam War in the 1960s would likely cause our model some distress.
#
# We could suggest to our employers that, if they are interested in discovering the causes of the cigarette sales trajectory, they execute a regression analysis in addition to the time series one.
