#Author: Taran Grove,
#Date: December 26, 2019

#Importing necessary libraries
import pandas as pd
import numpy as np
import math
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from numpy import linalg
from numpy import random
#


#Inverse chi squared (code originally written by Andrey Sarantsev)
def ichi2(degFreedom, scale):
    shape = degFreedom/2
    return ((shape*scale)/random.gamma(shape))
#

#Bayesian simple linear regression (code originally written by Andrey Sarantsev)
def BayesianRegression(x, y):
    n = len(x)
    M = [[np.dot(x, x), sum(x)], [sum(x), len(x)]]
    invM = linalg.inv(M)
    coeff = invM.dot([np.dot(x, y), sum(y)])
    slope = coeff[0]
    intercept = coeff[1]
    residuals = [y[k] - slope * x[k] - intercept for k in range(n)]
    var = np.dot(residuals, residuals)/(n-2)
    simVar = ichi2(n-2, var)
    simCoeff = random.multivariate_normal(coeff, invM*simVar)
    simSlope = simCoeff[0]
    simIntercept = simCoeff[1]
    return (simSlope, simIntercept, simVar)
#


#Code for computing linear regressions.
def regress(x, y):
    n = np.size(x)
    results = stats.linregress(x, y)
    residuals = np.array([y[k] - x[k] * results.slope - results.intercept for k in range(n)])
    stdDev = np.std(residuals)
    return [results.slope, results.intercept, stdDev, results.rvalue]
#


#Getting the data from "Bayesian.xlsx"
file = pd.ExcelFile('/Users/tarangrove/Documents/Research/Data/Bayesian.xlsx')
dataFrame1 = file.parse('StandardPoor')
dataFrame2 = file.parse('CPI')
dataFrame3 = file.parse('TreasuryBill')
dataFrame4 = file.parse('Shiller')
#


#Defining some constants
NSIMS = 10000
NYEARS = 84
HORIZON = 6
#


#Create 2D arrays for each worksheet
standardPoorArray = dataFrame1.values
consumerPriceIndexArray = dataFrame2.values
shillerArray = dataFrame4.values
#


#Make arrays for price, CPI, dividends, and earnings,
pricesArray = []
cpiArray = []
earningsArray  = []
dividendsArray = []

for i in range(NYEARS + 1):
    pricesArray.append(standardPoorArray[i][1])
    cpiArray.append(consumerPriceIndexArray[i][1])

for i in range(NYEARS):
    earningsArray.append(shillerArray[i][1])
    dividendsArray.append(shillerArray[i][2])
#


#Getting 2D array for earnings yield
#The two indices k and t will be associated to a particular earnings yield

earningsYieldArray = []
for k in range(11): #k will take values 3 through 10
    tempArray = []
    for t in range(NYEARS):
        #If k = 0, 1, 2, then do nothing
        if (0 <= k <= 2):
            tempArray.append(np.nan)
        #Else if k = 3, 4, ..., 10, compute earnings yield
        else:
            value = 0
            for i in range(k):
                value += earningsArray[t-i]
            value /= (k * pricesArray[t])
            if (k <= t+1):
                tempArray.append(value)
            else:
                tempArray.append(np.nan)
    earningsYieldArray.append(tempArray)
#


#Getting array for real returns (inflation adjusted) array
realReturnsArray = []
#Can only compute returns for t = 1, 2, ..., 84
for t in range(NYEARS):
    op1 = np.log((pricesArray[t+1] + dividendsArray[t]) / pricesArray[t])
    op2 = np.log(cpiArray[t+1] / cpiArray[t])
    realReturnsArray.append(op1 - op2)
#


#Code to check which value for k gives the
#best correlation between R(t) and E_k(t-1)
#This code ultimately shows k = 7 is optimal
for k in range(3, 11):
    prevEs = earningsYieldArray[k][k-1:NYEARS-1]
    currRs = realReturnsArray[k:NYEARS]
    print("For k = ", k, ", we have a correlation of ", regress(prevEs, currRs)[3])

print("")
print("Therefore, k = 7 is optimal.")
print("")
print("")
#
OPTIMAL = 7

#Since k = 7 is optimal, let us obtain the results of the regressions
prevEs = earningsYieldArray[OPTIMAL][OPTIMAL-1:NYEARS-1]
currEs = earningsYieldArray[OPTIMAL][OPTIMAL:NYEARS]
currRs = realReturnsArray[OPTIMAL:NYEARS]

#Using linear regression for simulations
EsResults = regress(prevEs, currEs)
RsResults = regress(prevEs, currRs)

EsSlope = EsResults[0]
EsIntercept = EsResults[1]
EsStdDev = EsResults[2]

RsSlope = RsResults[0]
RsIntercept = RsResults[1]
RsStdDev = RsResults[2]
#


#We now run the simulation using current values of E(t)
averageReturns1 = []
for simulation in range(NSIMS):
    returns = []
    currE = earningsYieldArray[OPTIMAL][len(earningsYieldArray[7]) - 1]
    currR = 0
    for t in range(HORIZON):
        delta = np.random.normal(0, EsStdDev)
        epsilon = np.random.normal(0, RsStdDev)
        currR = RsSlope * currE + RsIntercept + epsilon
        returns.append(currR)
        currE = EsSlope * currE + EsIntercept + delta
    averageReturns1.append(np.mean(returns))

print("Simulation 1 using current value of E(t) and best estimates from Linear Regression:")
print("Average of average returns over 20 years:", np.mean(averageReturns1))
print("Standard deviation of average returns over 20 years:", np.std(averageReturns1))
plt.plot(averageReturns1)
plt.show()
plt.hist(averageReturns1, bins = 100)
plt.show()
print("")
#


#We now run the simulation using long-term average values of E(t)
averageReturns2 = []
for simulatin in range(NSIMS):
    returns = []
    currE = np.mean(earningsYieldArray[OPTIMAL][OPTIMAL-1:NYEARS])
    currR = 0
    for t in range(HORIZON):
        delta = np.random.normal(0, EsStdDev)
        epsilon = np.random.normal(0, RsStdDev)
        currR = RsSlope * currE + RsIntercept + epsilon
        returns.append(currR)
        currE = EsSlope * currE + EsIntercept + delta
    averageReturns2.append(np.mean(returns))

print("Simulation 2 using average value of E(t) and best estimates from Linear Regression:")
print("Average of average returns over 20 years:", np.mean(averageReturns2))
print("Standard deviation of average returns over 20 years:", np.std(averageReturns2))
plt.plot(averageReturns1)
plt.show()
plt.hist(averageReturns2, bins = 100)
plt.show()
print("")
#


#We now run the BAYESIAN simulation using current values of E(t)
averageReturns1 = []
for simulation in range(NSIMS):
    returns = []
    currE = earningsYieldArray[OPTIMAL][len(earningsYieldArray[7]) - 1]
    currR = 0

    #Using BAYESIAN regression for simulations
    EsSlope, EsIntercept, EsVar = BayesianRegression(prevEs, currEs)
    EsStdDev = math.sqrt(EsVar)

    RsSlope, RsIntercept, RsVar = BayesianRegression(prevEs, currRs)
    RsStdDev = math.sqrt(RsVar)

    for t in range(HORIZON):
        delta = np.random.normal(0, EsStdDev)
        epsilon = np.random.normal(0, RsStdDev)
        currR = RsSlope * currE + RsIntercept + epsilon
        returns.append(currR)
        currE = EsSlope * currE + EsIntercept + delta
    averageReturns1.append(np.mean(returns))

print("Simulation 3 using current value of E(t) and Bayesian Regression:")
print("Average of average returns over 20 years:", np.mean(averageReturns1))
print("Standard deviation of average returns over 20 years:", np.std(averageReturns1))
plt.plot(averageReturns1)
plt.show()
plt.hist(averageReturns1, bins = 100)
plt.show()
print("")
#


#We now run the BAYESIAN simulation using long-term average values of E(t)
averageReturns2 = []
for simulatin in range(NSIMS):
    returns = []
    currE = np.mean(earningsYieldArray[OPTIMAL][OPTIMAL-1:NYEARS])
    currR = 0

    #Using BAYESIAN regression for simulations
    EsSlope, EsIntercept, EsVar = BayesianRegression(prevEs, currEs)
    EsStdDev = math.sqrt(EsVar)

    RsSlope, RsIntercept, RsVar = BayesianRegression(prevEs, currRs)
    RsStdDev = math.sqrt(RsVar)

    for t in range(HORIZON):
        delta = np.random.normal(0, EsStdDev)
        epsilon = np.random.normal(0, RsStdDev)
        currR = RsSlope * currE + RsIntercept + epsilon
        returns.append(currR)
        currE = EsSlope * currE + EsIntercept + delta
    averageReturns2.append(np.mean(returns))

print("Simulation 4 using average value of E(t) and Bayesian Regression:")
print("Average of average returns over 20 years:", np.mean(averageReturns2))
print("Standard deviation of average returns over 20 years:", np.std(averageReturns2))
plt.plot(averageReturns1)
plt.show()
plt.hist(averageReturns2, bins = 100)
plt.show()
print("")
#
