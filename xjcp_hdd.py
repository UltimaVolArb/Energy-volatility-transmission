# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 21:36:05 2023

@author: sigma
"""

import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

from statsmodels.tsa.stattools import grangercausalitytests

from scipy.stats import rankdata
price_data=pd.read_csv(r'E:\2023\XJCP Trader Quant\xjcp.csv')
df=pd.DataFrame(price_data)
print(df)


#for col in df.columns:
   # print(col)

# Replace zero in df with previous number
df = df.replace(0, method='ffill')
df


#Replace negatgive numbers with NaN
df[df < 0] = pd.np.nan


#Then forward fill Nan values
df = df.fillna(method='ffill')

#Remove negative number
#for col in df.columns:
   # df = df[df[col] >= 0]
    
#print('Modified Dataframe:\n', df)

#Fill missing number with previous number
#df.fillna(method='ffill', inplace=True)

#print(df)

#Then must remove NaN values
#df.dropna(inplace=True)

#print(df)

#df1=df.tail(-1)
#print(df1)

#Look up each column in the dataframe python
for column in df.columns:
    print(f"{column}: {df[column].values}")

# Plot time series by price for each asset
for col in df.columns[0:]:
    plt.plot(df[col])
    plt.xlabel('Time')
    plt.ylabel(col.capitalize())
    plt.title(f'{col.capitalize()} over time')
    plt.show()
    

#Example of naming the data in each column
#df1=df[df.columns[0]]
#df1

#Convert from price to return for stationary time series
daily_returns = df.apply(np.log).diff(1)
print(daily_returns)

#Plot
plt.style.use('fivethirtyeight')
daily_returns.plot(legend=0, figsize=(10,6), grid=True, title='Daily Equity Returns in the S&P500')
plt.tight_layout()


#daily_returns.dropna(inplace=True)

#Forward fill Nan values
daily_returns = daily_returns.fillna(method='ffill')
daily_returns

#Replace infinity with Nan
daily_returns=np.array(daily_returns)
daily_returns[~np.isfinite(daily_returns)] = np.nan

#Remove Nan values
daily_returns= daily_returns[~np.isnan(daily_returns)]

#Normality test
from scipy.stats import shapiro
# Perform Shapiro-Wilk normality test
stat, p = shapiro(daily_returns)

# Interpret results
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
    
    
#Reshape 1D array into 2D array
daily_returns= daily_returns.reshape(-1, 1)

#standarized data
standard_returns=(daily_returns-daily_returns.mean())/daily_returns.std()
standard_returns_PCA=pd.DataFrame(standard_returns)



#Perform Principal Component Analysis (PCA)
pca=PCA()
pca.fit(standard_returns_PCA)
#Get eigenvector loadings
loadings=pca.components_
loadings
print(loadings) #something is wrong here


#Another way


#Calculate first eivenvector loadings
#weights = abs(first_pc)/sum(abs(first_pc))
#weighted_daily_returns = (weights*daily_returns).sum(1)
#weighted_daily_returns.cumsum().apply(np.exp).plot()



#Convert back to 1D array for daily returns
price_data2=pd.read_csv(r'E:\2023\XJCP Trader Quant\xjcp.csv')
df2=pd.DataFrame(price_data2)

# Replace zero in df with previous number
df2 = df.replace(0, method='ffill')
df2


#Replace negatgive numbers with NaN
df2[df2 < 0] = pd.np.nan


#Then forward fill Nan values
df2 = df2.fillna(method='ffill')

#************exp 2 for PCA with first difference****

#Principal Components Analsyis (PCA) for dimension reduction
#Use first differenced data
diff_ = df.diff(-1)
diff_.dropna(inplace=True)
diff_.tail()
#Derive volatility
#The drift of forward rate is fully determined by volatility of forward rate dynamics

#Covariance
cov_= pd.DataFrame(np.cov(diff_, rowvar=False)*252/10000, columns=diff_.columns, index=diff_.columns)
cov_.style.format("{:.4%}")


#Eigen 
# Perform eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_)

# Sort values
idx = eigenvalues.argsort()[::-1]   
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

# Format into a DataFrame 
df_eigval = pd.DataFrame({"Eigenvalues": eigenvalues})

eigenvalues

#Explained variance
# Work out explained proportion 
df_eigval["Explained proportion"] = df_eigval["Eigenvalues"] / np.sum(df_eigval["Eigenvalues"])
df_eigval = df_eigval[:10]
df_eigval

#Format as percentage
df_eigval.style.format({"Explained proportion": "{:.2%}"})



# Subsume first 3 components into a dataframe
pcadf = pd.DataFrame(eigenvectors[:,0:3], columns=['PC1','PC2','PC3'])
pcadf[:10]


#Convergence of eivenvectors
plt.plot(pcadf[1:30])


#Time series analysis VECM********************************#          
# Perform Johansen test



#Multivariate Time Series Analysis by Vector Error Correction Model (VECM)

subset=df.iloc[3000:6000,39:44]
subset
print(subset)
#model = VAR(subset)
#print(model)

from statsmodels.tsa.vector_ar.vecm import coint_johansen
cointresult = coint_johansen(subset, det_order=0, k_ar_diff=1)

# Get the results
cv = cointresult.cvm
p_values = 1 - cv[:, 1]
print(p_values)
#data = np.log(df).diff().dropna()
#model= sm.tsa.VECM(ndata, k_ar_diff=1, coint_rank=1, deterministic='nc')
#model= VECM(df, k_ar_diff=1, coint_rank=1, deterministic='nc')
diffsubdata = subset.diff(-1)
diffsubdata.dropna(inplace=True)
diffsubdata.tail()

#Stationary check
from statsmodels.tsa.stattools import adfuller

def stationarity(data, cutoff=0.05):
   if adfuller(data)[1] < cutoff:
        print('The series is stationary')
        print('p-value = ', adfuller(data)[1])
   else:
        print('The series is NOT stationary')
        print('p-value = ', adfuller(data)[1])

d39=df.iloc[3000:6000,39]
d40=df.iloc[3000:6000,40]
d41=df.iloc[3000:6000,41]
d42=df.iloc[3000:6000,42]
d43=df.iloc[3000:6000,43]
d44=df.iloc[3000:6000,44]
stationarity(d39)
stationarity(d40)
stationarity(d41)
stationarity(d42)
stationarity(d43)
stationarity(d44)        
#stationarity(diffsubdata)    
# Perform PP test
#unitroot_result = adfuller(diffsubdata)
# Get the test statistic and p-value
#print('ADF statistic: ', unitroot_result[0])
#print('p-value: ', unitroot_result[1])



#VAR model begins
model= VAR(diffsubdata)
results = model.fit()
results.summary()

results.plot()

results.plot_acorr()

model.select_order(15)

results = model.fit(maxlags=15, ic='aic')

lag_order = results.k_ar

results.plot_forecast(10)

#Impulse response
irf = results.irf(10)

irf.plot(orth=False)

irf.plot_cum_effects(orth=False)

fevd = results.fevd(5)

#Spread and mean reversion
#half life

def estimate_half_life(spread):
    x=spread.shift().iloc[1:].to_frame().assign(const=1)
    y=spread.diff().iloc[1:]
    beta=(np.linalg.inv(x.T@x)@x.T@y).iloc[0]
    halflife=int(round(-np.log(2)/beta,0))
    return max(halflife,1)

#Best pair for pair trading
from scipy.stats import pearsonr
# Find the best pair simply by looking at correlations
corr_matrix = diff_.corr()

# Find the pair with the highest correlation coefficient
pairs = [(corr_matrix.iloc[i, j], corr_matrix.columns[i], corr_matrix.columns[j]) for i in range(len(corr_matrix.columns)) for j in range(i+1, len(corr_matrix.columns))]
best_pair = max(pairs)

# Output the best pair
print('Best pair for pair trading:', best_pair[1], 'and', best_pair[2])

#from HalfLife import estimate_half_life

a=df.iloc[7:1007,19]
a
b=df.iloc[7:1007,28]
b
spread=a-b
estimate_half_life(spread)
plt.plot(spread)
plt.title("Spread between assets 20 and 29 from day 7 to 1006")


#Best pair for divergence trading
best_pair_dir = min(pairs)

# Output the best pair for divergence trading
print('Best pair for divergence trading:', best_pair_dir[1], 'and', best_pair_dir[2])
#Best pair for divergence trading: 54 and 55

c=df.iloc[7:1007,53]
d=df.iloc[7:1007,54]
spread_dir=c-d



#Univaraite forecast
#Let's say we want to forecast the spread
# Fit an ARIMA model to the data
from statsmodels.tsa.arima.model import ARIMA
ARIMAmodel = ARIMA(spread, order=(1, 1, 1))
ARIMAmodel_fit = ARIMAmodel.fit()

# Make a forecast for the next 50 period
ARIMAforecast = ARIMAmodel_fit.forecast(50)
ARIMAforecast
plt.plot(spread, label='Original Data')
plt.plot(ARIMAforecast, label='ARIMA Forecast')
plt.legend()
plt.show()

#Volatility analysis
# Randomly select 1 asset from the DataFrame for volatility analysis
import random
random_asset = df.columns[random.randint(0, len(df.columns)-1)]
#print(random_asset)
from arch import arch_model

#Assume it randomly selects asset 50
A50=df.iloc[4002:5002,49]

A50_returns = np.log(A50).diff().fillna(0)

# Visualize the selected asset's daily returns
plt.plot(A50_returns, color='lightpink')
plt.title('Asset 50 Returns')
plt.grid(True)

#Skew and kurtosis
from scipy.stats import skew, kurtosis
skewness = skew(A50_returns)
kurt = kurtosis(A50_returns)

# Print results
print('Skewness:', skewness)
print('Kurtosis:', kurt)

# Perform Shapiro-Wilk normality test
stat, p = shapiro(A50_returns)

# Interpret results
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

#5-Day Historical volatility of the selected asset
HV_5D=A50_returns.rolling(5).std()
HV_5D=HV_5D.fillna(method='ffill')
plt.plot(HV_5D, color='purple')


HV_5D_stat=HV_5D.describe()
HV_5D_stat



#Exponential GARCH (1,1,1) that captures asymmetric news shock
am = arch_model(A50_returns, vol="Garch", p=1, o=1, q=1, dist="Normal")
res = am.fit(update_freq=5)
vforecasts = res.forecast(reindex=False)
print(vforecasts.mean.iloc[-3:])
print(vforecasts.residual_variance.iloc[-3:])
print(vforecasts.variance.iloc[-3:])
#Volatility forecast in the next five days
vforecasts = res.forecast(horizon=5, reindex=False)
print(vforecasts.residual_variance.iloc[-3:])

#Derivatives pricing
from scipy import sparse
#pip install quantecon
import quantecon as qe
from quantecon.markov import DiscreteDP, backward_induction, sa_indices


A50_price=df.iloc[6432,49]

#Options pricing model inputs
T = 0.08       # Time expiration (years)
vol = 1.06    # Annual volatility
r = 0.039     # Annual interest rate
strike = A50_price+200  # Strike price
p0 =A50_price        # Current price
N = 20       # Number of periods to expiration

# Time length of a period
tau = T/N
# Discount factor
beta = np.exp(-r*tau)
# Up-jump factor
u = np.exp(vol*np.sqrt(tau))
# Up-jump probability
q = 1/2 + np.sqrt(tau)*(r - (vol**2)/2)/(2*vol)
# Possible price values
ps = u**np.arange(-N, N+1) * p0
# Number of states
n = len(ps) + 1  # State n-1: "the option has been exercised"
# Number of actions
m = 2  # 0: hold, 1: exercise
# Number of feasible state-action pairs
L = n*m - 1  # At state n-1, there is only one action "do nothing"
# Arrays of state and action indices
s_indices, a_indices = sa_indices(n, m)
s_indices, a_indices = s_indices[:-1], a_indices[:-1]
# Reward vector
R = np.empty((n, m))
R[:, 0] = 0
R[:-1, 1] = strike - ps
R = R.ravel()[:-1]

# Transition probability array
Q = sparse.lil_matrix((L, n))
for i in range(L-1):
    if a_indices[i] == 0:
        Q[i, min(s_indices[i]+1, len(ps)-1)] = q
        Q[i, max(s_indices[i]-1, 0)] = 1 - q
    else:
        Q[i, n-1] = 1
Q[L-1, n-1] = 1

# Put options optimal exercise boundary
ddp = DiscreteDP(R, Q, beta, s_indices, a_indices)

vs, sigmas = backward_induction(ddp, N)

v = vs[0]
max_exercise_price = ps[sigmas[::-1].sum(-1)-1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot([0, strike], [strike, 0], 'k--')
axes[0].plot(ps, v[:-1])
axes[0].set_xlim(0, strike*2)
axes[0].set_xticks(np.linspace(0, 4, 5, endpoint=True))
axes[0].set_ylim(0, strike)
axes[0].set_yticks(np.linspace(0, 2, 5, endpoint=True))
axes[0].set_xlabel('Asset Price')
axes[0].set_ylabel('Premium')
axes[0].set_title('Put Option Value')

axes[1].plot(np.linspace(0, T, N), max_exercise_price)
axes[1].set_xlim(0, T)
axes[1].set_ylim(1.6, strike)
axes[1].set_xlabel('Time to Maturity')
axes[1].set_ylabel('Asset Price')
axes[1].set_title('Put Option Optimal Exercise Boundary')
axes[1].tick_params(right='on')

plt.show()



#Implied Volatility
# Data Manipulation
import pandas as pd
from numpy import *
from datetime import timedelta
#import yfinance as yf
from tabulate import tabulate

# Math & Optimization
from scipy.stats import norm
from scipy.optimize import fsolve

# Plotting
#import matplotlib.pyplot as plt
#import cufflinks as cf
#cf.set_config_file(offline=True)



class BS:
    
    """
    This is a class for Options contract for pricing European options on stocks/index without dividends.
    
    Attributes: 
        spot          : int or float
        strike        : int or float 
        rate          : float
        dte           : int or float [days to expiration in number of years]
        volatility    : float
        callprice     : int or float [default None]
        putprice      : int or float [default None]
    """    
    
    def __init__(self, spot, strike, rate, dte, volatility, callprice=None, putprice=None):
        
        # Spot Price
        self.spot = spot
        
        # Option Strike
        self.strike = strike
        
        # Interest Rate
        self.rate = rate
        
        # Days To Expiration
        self.dte = dte
        
        # Volatlity
        self.volatility = volatility
        
        # Callprice # mkt price
        self.callprice = callprice
        
        # Putprice # mkt price
        self.putprice = putprice
            
        # Utility 
        self._a_ = self.volatility * self.dte**0.5
        
        if self.strike == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            self._d1_ = (log(self.spot / self.strike) + \
                     (self.rate + (self.volatility**2) / 2) * self.dte) / self._a_
        
        self._d2_ = self._d1_ - self._a_
        
        self._b_ = e**-(self.rate * self.dte)
        
        
        # The __dict__ attribute
        '''
        Contains all the attributes defined for the object itself. It maps the attribute name to its value.
        '''
        for i in ['callPrice', 'putPrice', 'callDelta', 'putDelta', 'callTheta', 'putTheta', \
                  'callRho', 'putRho', 'vega', 'gamma', 'impvol']:
            self.__dict__[i] = None
        
        [self.callPrice, self.putPrice] = self._price()
        [self.callDelta, self.putDelta] = self._delta()
        [self.callTheta, self.putTheta] = self._theta()
        [self.callRho, self.putRho] = self._rho()
        self.vega = self._vega()
        self.gamma = self._gamma()
        self.impvol = self._impvol()
    
    # Option Price
    def _price(self):
        '''Returns the option price: [Call price, Put price]'''

        if self.volatility == 0 or self.dte == 0:
            call = maximum(0.0, self.spot - self.strike)
            put = maximum(0.0, self.strike - self.spot)
        else:
            call = self.spot * norm.cdf(self._d1_) - self.strike * e**(-self.rate * \
                                                                       self.dte) * norm.cdf(self._d2_)

            put = self.strike * e**(-self.rate * self.dte) * norm.cdf(-self._d2_) - \
                                                                        self.spot * norm.cdf(-self._d1_)
        return [call, put]

    # Option Delta
    def _delta(self):
        '''Returns the option delta: [Call delta, Put delta]'''

        if self.volatility == 0 or self.dte == 0:
            call = 1.0 if self.spot > self.strike else 0.0
            put = -1.0 if self.spot < self.strike else 0.0
        else:
            call = norm.cdf(self._d1_)
            put = -norm.cdf(-self._d1_)
        return [call, put]

    # Option Gamma
    def _gamma(self):
        '''Returns the option gamma'''
        return norm.pdf(self._d1_) / (self.spot * self._a_)

    # Option Vega
    def _vega(self):
        '''Returns the option vega'''
        if self.volatility == 0 or self.dte == 0:
            return 0.0
        else:
            return self.spot * norm.pdf(self._d1_) * self.dte**0.5 / 100

    # Option Theta
    def _theta(self):
        '''Returns the option theta: [Call theta, Put theta]'''
        call = -self.spot * norm.pdf(self._d1_) * self.volatility / (2 * self.dte**0.5) - self.rate * self.strike * self._b_ * norm.cdf(self._d2_)

        put = -self.spot * norm.pdf(self._d1_) * self.volatility / (2 * self.dte**0.5) + self.rate * self.strike * self._b_ * norm.cdf(-self._d2_)
        return [call / 365, put / 365]

    # Option Rho
    def _rho(self):
        '''Returns the option rho: [Call rho, Put rho]'''
        call = self.strike * self.dte * self._b_ * norm.cdf(self._d2_) / 100
        put = -self.strike * self.dte * self._b_ * norm.cdf(-self._d2_) / 100

        return [call, put]
    
    # Option Implied Volatility
    def _impvol(self):
        '''Returns the option implied volatility'''
        if (self.callprice or self.putprice) is None:
            return self.volatility
        else:
            def f(sigma):
                option = BS(self.spot,self.strike,self.rate,self.dte,sigma)
                if self.callprice:
                    return option.callPrice - self.callprice # f(x) = BS_Call - MarketPrice
                if self.putprice and not self.callprice:
                    return option.putPrice - self.putprice

            return maximum(1e-5, fsolve(f, 0.2)[0])
        
# Initialize option
#from BS import BS
option = BS(p0 ,strike,r,20/250,1.405,500)

header = ['Option Price', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'IV']
table = [[option.callPrice, option.callDelta, option.gamma, option.callTheta, option.vega, option.callRho, option.impvol]]

print(tabulate(table,header))

# Bisection Method

def bisection_iv(className, spot, strike, rate, dte, volatility, callprice=None, putprice=None, high=500.0, low=0.0):
    
    if callprice:
        price = callprice
    if putprice and not callprice:
        price = putprice
        
    tolerance = 1e-7
        
    for i in range(10000):
        mid = (high + low) / 2 # c= (a+b)/2
        if mid < tolerance:
            mid = tolerance
            
        if callprice:
            estimate = eval(className)(spot, strike, rate, dte, mid).callPrice # Blackscholes price
        if putprice:
            estimate = eval(className)(spot, strike, rate, dte, mid).putPrice
        
        if round(estimate,6) == price:
            break
        elif estimate > price: 
            high = mid # b = c
        elif estimate < price: 
            low = mid # a = c
    
    return mid



bisection_iv('BS',p0 ,strike,r,20/250,1.405,callprice=300)
             
bisection_iv('BS',p0 ,strike,r,20/250,1.405,putprice=550)

