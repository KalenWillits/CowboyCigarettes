# %% markdown
# # Springboard Time Series - 'Cowboy Cigarettes' Case Study - Tier 2
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
import _ _ _ as pd
import numpy as _ _ _
import _ _ _.pylab as plt
%matplotlib inline
# %% markdown
# ### 1b. Load the data
# Call the variable `cigData`.
# %% codecell
_ _ _
# %% markdown
# ### 1c. Explore the data
# We now need to check whether the data conduces to a time series style analysis.
# %% codecell
_ _ _
# %% markdown
# Over a million cigarettes sold in the month of January 1949. This certainly is a popular cigarette brand.
# %% markdown
# Check out the columns feature of the data. How many columns are there?
# %% codecell
_ _ _
# %% markdown
# Let's check out the data types of our columns.
# %% codecell
_ _ _.dtypes
# %% markdown
# Check whether there are any null values.
# %% codecell
_ _ _.isnull().values.any()
# %% markdown
# ## 2. Cleaning, transforming and visualizing
# %% markdown
# ### 2a. Dropping unwanted columns
# We need to cut that `Unnamed: 0` column. Delete it here.
# %% codecell
_ _ _
# %% markdown
# ### 2b. Nomenclature
# %% markdown
# We can see that the `Time` column actually has the granularity of months. Change the name of that column to `Month`.
# %% codecell
_ _ _
# %% markdown
# Call a head() to check this has worked.
# %% codecell
_ _ _
# %% codecell
_ _ _
# %% markdown
# ### 2c. Type conversions
# %% markdown
# Now, do time series analysis on a Pandas dataframe is overkill, and is actually counter-productive. It's much more easy to carry out this type of analysis if we convert our data to a series first.
#
# Notice that the `Month` field was an object. Let's type convert the `Month` column to a Python `datetime`, before making that the index.
# %% codecell
_ _ _['Month'] = pd._ _ _(cigData['Month'])
_ _ _._ _ _('Month', inplace = True)
# %% markdown
# Perfect!
# %% markdown
# ### 2d. Making a predictor variable `y`
# %% markdown
# The data is now indexed by date, as time series data ought to be.
#
# Since we want to predict the number of cigarette sales at Cowboy cigarettes, and `y` is typically used to signify a predictor variable, let's create a new variable called `y` and assign the indexed #Passenger column.
# %% codecell
y = _ _ _['#CigSales']
# %% markdown
# Check the type of our new variable.
# %% codecell
type(_ _ _)
# %% markdown
# ### 2e. Getting summary statistics for `y`
# %% markdown
# Get the summary statistics of our data here.
# %% codecell
_ _ _.describe()
# %% markdown
# Try visualizing the data. A simple `matplotlib` plot should do the trick.
# %% markdown
# ### 2f. Plotting `y`
# %% codecell
y._ _ _()
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
from statsmodels.tsa.seasonal import _ _ _

# Make a variable called decomposition, and assign it y passed to seasonal_decompose()
_ _ _ = seasonal_decompose(y)

# Make three variables for trend, seasonal and residual components respectively.
# Assign them the relevant features of decomposition
trend = decomposition.trend
seasonal = _ _ _.seasonal
_ _ _ = decomposition.resid

# Plot the original data, the trend, the seasonality, and the residuals
plt.subplot(411)
plt.plot(y, label = '_ _ _')
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(trend, label = 'Trend')
plt.legend(loc = 'best')
plt._ _ _(413)
plt._ _ _(seasonal, label = '_ _ _')
plt._ _ _(loc = 'best')
plt._ _ _(414)
plt._ _ _(residual, label = 'Residuals')
plt._ _ _(loc = 'best')
plt._ _ _()
# %% markdown
# ### 3b. Testing for stationarity with KPSS
# As you know, when doing time series analysis we always have to check for stationarity. Imprecisely, a time series dataset is stationary just if its statistical features don't change over time. A little more precisely, a stationary time series dataset will have constant mean, variance, and covariance.
#
# There are many ways to test for stationarity, but one of the most common is the KPSS test. The Null hypothesis of this test is that the time series data in question is stationary; hence, if the *p*-value is less than the significance level (typically 0.05, but we decide) then we reject the Null and infer that the data is not stationary.
# %% codecell
from statsmodels.tsa.stattools import kpss

# Use kpss()
_ _ _(y)
# %% markdown
# Since our p-value is less than 0.05, we should reject the Null hypothesis and deduce the non-stationarity of our data.
#
# But our data need to be stationary! So we need to do some transforming.
# %% markdown
# ### 3c. Making the data stationary
# Let's recall what it looks like.
# %% codecell
y._ _ _()
# %% markdown
# In our plot, we can see that both the mean and the variance *increase as time progresses*. At the moment, our data has neither a constant mean, nor a constant variance (the covariance, however, seems constant).
#
# One ofte  used way of getting rid of changing variance is to take the natural log of all the values in our dataset. Let's do this now.
# %% codecell
# Declare a variable called y_log
_ _ _ = np.log(y)
# %% markdown
#
# When you plot this, you can see how the variance in our data now remains contant over time.
# %% codecell
y_log._ _ _()
# %% markdown
# We now have a constant variance, but we also need a constant mean.
#
# We can do this by *differencing* our data. We difference a time series dataset when we create a new time series comprising the difference between the values of our existing dataset.
#
# Python is powerful, and we can use the `diff()` function to do this. You'll notice there's one less value than our existing dataset (since we're taking the difference between the existing values).
# %% codecell
kpss(y_log._ _ _().dropna())
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
from sklearn.metrics import _ _ _
from statsmodels.tsa.arima_model import _ _ _
# %% markdown
# #### 3di. Make a function to find the MSE of a single ARIMA model
# %% codecell
# Make a function called evaluate_arima_model to find the MSE of a single ARIMA model
def _ _ _(data, arima_order):
    # Needs to be an integer because it is later used as an index.
    # Use int()
    split=_ _ _(len(data) * 0.8)
    # Make train and test variables, with 'train, test'
    _ _ _, _ _ _ = data[0:split], data[split:len(data)]
    past=[x for x in train]
    # make predictions. Declare a variable with that name
    _ _ _ = list()
    for i in range(len(test)):#timestep-wise comparison between test data and one-step prediction ARIMA model.
        model = ARIMA(past, order=arima_order)
        model_fit = model.fit(disp=0)
        future = model_fit.forecast()[0]
        # Append() here
        predictions._ _ _(future)
        past.append(test[i])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    # Return the error
    return _ _ _
# %% markdown
# #### 3dii. Make a function to evaluate the different ARIMA models with different p, d, and q values
# %% codecell
# Make a function called evaluate_models to evaluate different ARIMA models with several different p, d, and q values.
def _ _ _(dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    # Iterate through p_values
    for p in _ _ _:
        # Iterate through d_values
        for d in _ _ _:
            # Iterate through q_values
            for q in _ _ _:
                # p, d, q iterator variables in that order
                order = (_ _ _,_ _ _,_ _ _)
                try:
                    # Make a variable called mse for the Mean squared error
                    _ _ _ = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    return print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
# %% codecell
# Now, we choose a couple of values to try for each parameter: p_values, d_values and q_values
# Fill in the blanks as appropriate
_ _ _ = [x for x in range(0, 3)]
_ _ _ = [_ _ _]
_ _ _ = [_ _ _]
# %% codecell
# Finally, we can find the optimum ARIMA model for our data.
# Nb. this can take a while...!
import warnings
warnings.filterwarnings("ignore")
evaluate_models(y_log, p_values, d_values, q_values)
# %% markdown
# So the best p,d, q, parameters for our ARIMA model are 2, 1, 1 respectively. Now we know this, we can build the model.
# %% codecell
p=_ _ _
_ _ _=_ _ _
q=_ _ _
model = ARIMA(y_log, order=(p,d,q))
model_fit = model.fit()
forecast = model_fit.forecast(24)
# %% markdown
# We can take a look at a summary of the model this library has built around our data.
# %% codecell
# Call summary() on model_fit
model_fit._ _ _
# %% markdown
# ### 3e. Visualize the results
#
# Visualize the original dataset plotted against our model.
# %% codecell
# Call figure() and plot() on the plt
plt._ _ _(figsize=(15,10))
plt._ _ _(y_log.diff())
plt._ _ _(model_fit.predict(), color = 'red')
# %% markdown
# ### 3f. Application: Forecasting
#
# We've done well: our model fits pretty closely to our existing data. Let's now use it to forecast what's likely to occur in future.
# %% codecell
# Declare a variable called forecast_period with the amount of months to forecast, and
# create a range of future dates that is the length of the periods you've chosen to forecast
_ _ _ = _ _ _
date_range = pd.date_range(y_log.index[-1], periods = forecast_period,
              freq='MS').strftime("%Y-%m-%d").tolist()

# Convert that range into a dataframe that includes your predictions
# First, call DataFrame on pd
future_months = pd._ _ _(date_range, columns = ['Month']
# Let's now convert the 'Month' column to a datetime object with to_datetime
future_months['Month'] = pd._ _ _(future_months['Month'])
future_months.set_index('Month', inplace = True)
future_months['Prediction'] = forecast[0]

# Plot your future predictions
# Call figure() on plt
plt._ _ _(figsize=(15,10))
plt.plot(y_log)
plt.plot(y_log['Nov 1960'].append(future_months['Prediction']))
plt.show()
# %% codecell
# Now plot the original variable y
# Use the same functions as before
plt._ _ _(figsize=(15,10))
plt._ _ _(y)
plt._ _ _(np.exp(y_log['Nov 1960'].append(future_months['Prediction'])))
plt._ _ _()
# %% markdown
# ## 4. Evaluating and Concluding
#
# Our model captures the centre of a line that's increasing at a remarkable rate. Cowboy Cigarettes sell more cigarettes in the summer, perhaps due to the good weather, disposable income and time off that people enjoy, and the least in the winter, when people might be spending less and enjoying less free time outdoors.
#
# Remarkably, our ARIMA model made predictions using just one variable. We can only speculate, however, on the causes of the behaviour predicted by our model. We should also take heed that spikes in data, due to sudden unusual circumstances like wars, are not handled well by ARIMA; and the outbreak of the Vietnam War in the 1960s would likely cause our model some distress.
#
# We could suggest to our employers that, if they are interested in discovering the causes of the cigarette sales trajectory, they execute a regression analysis in addition to the time series one.
