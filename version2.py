#Created by Michael Reyes and Andrey Sarantsev
#University of Nevada in Reno
#Department of Mathematics and Statistics
#Finished March 30, 2020

import numpy
from numpy import random
from numpy import linalg
import math
from scipy import stats
import pandas
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from statsmodels import api

def ichi2(df, scale):
    shape = df/2
    return ((shape*scale)/numpy.random.gamma(shape))

# Bayesian Regression
def BayesSimMulti(coeff, vhat, gramMatrix_inverse, df):
    simVar = ichi2(df, vhat) # Simulated variance
    simCoeff = numpy.random.multivariate_normal(coeff, simVar*gramMatrix_inverse) # Sim coefficient
    return (simVar, simCoeff)

def BayesSim(coeff, vhat, gramMatrix_inverse, df):
    simVar = ichi2(df, vhat) # Simulated variance
    simCoeff = numpy.random.normal(coeff, simVar*gramMatrix_inverse) # Sim coefficient
    return (simVar, simCoeff)
# Here we read the data 
# and do preliminary preparation

df1 = pandas.read_excel('Data.xlsx', sheet_name = 'Annual')
df2 = pandas.read_excel('Data.xlsx', sheet_name = 'Monthly')
data1 = df1.values
data2 = df2.values

# First we read annual data
# index, first month of year, daily average, annual
Price = data1[:, 1].astype(float)  
# This number is 2020-1871 = 149.
ALLYEARS = len(Price) 
# S&P dividend paid last year per share (nominal)
Dividend = data1[:ALLYEARS, 2].astype(float)
# S&P earnings per share last year (nominal)
Earnings = data1[:ALLYEARS, 3].astype(float)  

# Then we read monthly data and annualize it
# Consumer Price Index monthly
CPI = data2[:, 2].astype(float) 
#2034-1871 = 63: How late does monthly data start
Index = data2[:, 1].astype(float)
#Annual March CPI data
cpi = CPI[11::12]
#April 1 prices
index = Index[11::12]
FIRSTYEAR = 75 #Our first year of observations, corresponds to 1946 = 1871 + 75
GAP = 64 #Indication that monthly data starts from 1935 = 1871 + 64
index = index[FIRSTYEAR - GAP:] #Index level at beginning of year, starting from 1946
cpi = cpi[FIRSTYEAR - GAP:] #Consumer Price Index
NYEARS = len(cpi) - 1 #Number of years in our data, 2019-1946 = 73
dividend = Dividend[FIRSTYEAR:-1] #Annual dividends, starting from 1946
earnings = Earnings[FIRSTYEAR:-1] #Annual earnings, starting from 1946
allYrs = range(1871 + FIRSTYEAR, 1871 + FIRSTYEAR + NYEARS)
print('annual dividends')
pyplot.plot(allYrs, dividend)
pyplot.show()
print('annual earnings')
pyplot.plot(allYrs, earnings)
pyplot.show()
print('annual index level')
pyplot.plot(range(1871 + FIRSTYEAR, 1871 + FIRSTYEAR + NYEARS + 1), index)
pyplot.show()

# Inflation rate
inflation = numpy.array([math.log(cpi[k+1]/cpi[k]) for k in range(NYEARS)])
print('inflation')
pyplot.plot(inflation)
pyplot.show()
# Total annual (nominal) return for S&P including dividends
TRStock = numpy.array([math.log(index[k + 1] + dividend[k]) - math.log(index[k]) for k in range(NYEARS)])
print('We start from year ', 1871 + FIRSTYEAR)
print('having in total years ', NYEARS)
print('Annual total nominal stock returns')
pyplot.plot(TRStock)
pyplot.show()
# Now we aggregate our data in 3-year steps
# From now on, we work with such aggregated data
# How many years in time step, to make normality assumption true
print('Now we create and work with aggregated data')
STEP = 3
# How many steps
NSTEPS = int(NYEARS / STEP)  
print('Step size is ', STEP)
print('How many steps? ', NSTEPS)
print('')
Years = range(NSTEPS)
#Time horizon = Number of steps for simulation
#Multiply by M = 3 to get the number of years
HORIZONSIM = 10
NSIMS = 10000 # Number of simulations
# Cumulative nominal total return for each time step for S&P
NominalReturn = numpy.array([sum(TRStock[STEP*k:STEP*k + STEP]) for k in range(NSTEPS)])
# Cumulative nominal earnings per share for S&P for each time step
EarnCum = numpy.array([sum(earnings[STEP*k:STEP*k + STEP]) for k in range(NSTEPS)])
print('Graph of summed earnings')
pyplot.plot(Years, [math.log(item) for item in EarnCum])
pyplot.show()
# Inflation in few years
Inflation = numpy.array([sum(inflation[STEP*k:STEP*k+STEP]) for k in range(NSTEPS)])
print('Inflation')
pyplot.plot(Years, Inflation)
pyplot.show()
# Real total stock market returns
RealReturn = NominalReturn - Inflation
print('Real Return')
pyplot.plot(Years, RealReturn)
pyplot.show()
#Earnings growth
EG = numpy.array([math.log(EarnCum[k+1]/EarnCum[k]) for k in range(NSTEPS-1)])
#Deviation
Deviation = NominalReturn[1:] - EG
print('Deviation is implied dividend yield')
print('Because this is total return minus earnings growth')
print('If stock prices grew proportionally to earnings')
print('then this would be exactly dividend yield')
print('It is usually positive, but how much on average?')
print('We need to subtract implicit average')
print('to find whether the market is hot or cold')
print('Equivalently, detrend cumulative sum of prior deviations')
print('Hot = cumulative detrended deviation is high')
print('Cold = vice versa')
print('For example, if average deviation is 3%')
print('and recently it was systematically above 3%')
print('then the market is overheated')
pyplot.plot(Deviation)
pyplot.show()
meanDev = numpy.mean(Deviation)
stdDev = numpy.std(Deviation)
print('mean for deviation = ', meanDev)
print('stdev for deviation = ', stdDev)
#print('Shapiro-Wilk p = ', stats.shapiro(Deviation)[1])
#print('Jarque-Bera p = ', stats.jarque_bera(Deviation)[1])
#qqplot(Deviation, line = 's')
#pyplot.show()
#Heat
Heat = numpy.append([0], numpy.cumsum(Deviation))
print('Heat is the cumsum of Deviation')
print('This measures how overpriced is the market')
print('But we will later detrend it')
print('We take coefficients a and m and regress')
print('deviation at step t upon a*(heat(t-1) - m*(t-1))')
print('We get regression coefficients a and -a*m')
print('From there we infer m, the true average implied dividend yield')
pyplot.plot(Heat)
pyplot.show()
#Real earnings growth, adjusted for inflation
REG = EG - Inflation[1:]
print('Real Earnings Growth')
pyplot.plot(REG)
pyplot.show()
meanREG = numpy.mean(REG)
stdREG = numpy.std(REG)
print('mean REG = ', meanREG)
print('std REG = ', stdREG)
print('Shapiro-Wilk p = ', stats.shapiro(REG)[1])
print('Jarque-Bera p = ', stats.jarque_bera(REG)[1])
qqplot(REG, line = 's')
pyplot.show()
print('This shows that REG is i.i.d. normal')
print('Main Regression')
U = numpy.ones(NSTEPS-1)
print('Regression with detrended heat, REG')
print('We initially wanted to do regression without REG')
print('But found that residuals and REG were correlated')
print('This would force us to use multidimensional Bayesian inference')
print('This is cumbersome, so I just added REG into regression')
print('This way we can separately simulate REG and residuals')
Y = pandas.DataFrame({'Const': U, 'REG': REG, 'Heat': Heat[:-1], 'Trend': range(NSTEPS-1)})
Reg = api.OLS(Deviation, Y).fit()
print(Reg.summary())
intercept = Reg.params[0]
heatcoeff = Reg.params[2]
trendcoeff = Reg.params[3]
REGcoeff = Reg.params[1]
IDY = trendcoeff/abs(heatcoeff)
print('Trend coefficient is', IDY)
print('Divide by', STEP)
print('and get implied dividend yield annualized:', IDY/STEP)
Residuals = Deviation - Reg.predict(Y)
stderr = math.sqrt((1/(NSTEPS-5))*numpy.dot(Residuals, Residuals))
print('standard error residuals = ', stderr)
qqplot(Residuals, line = 's')
pyplot.show()
print('Normality testing of Residuals')
print('Shapiro-Wilk p = ', stats.shapiro(Residuals)[1])
print('Jarque-Bera p = ', stats.jarque_bera(Residuals)[1])
print('Then we can simulate real earnings growth as i.i.d. normal')
print('with mean and stdev computed')
print('Using Bayesian normal distribution')
print('')
print('And simulate main regression using Bayesian inference')
print('We simulate these REG variables and regression independently')
print('and plug REG as a factor in regression')
print('Sum Deviation and REG, get total real return for', STEP, 'years')
print('')
print('We regressed our deviation upon REG to make correlation')
print('between residuals and REG to be zero')
print('so that we can simulate Bayesian posteriors independently')
print('Note that p value for Heat is 0.034, thus Heat is significant')
print('Try 10000 simulations for', HORIZONSIM*STEP, 'years', HORIZONSIM, 'time steps')
print('For each sim, find A = average annualized total real return')
print('which is the sum of deviation and REG')
print('summed over these', HORIZONSIM, 'steps, and divided by', HORIZONSIM*STEP)
print('Here we want to study influence of only heat')
print('and emphasize that it is statistically significant')
print('I chose starting year', 1871+FIRSTYEAR, 'so that all QQ plots are normal')

#Set up for Bayesian Inference
#Set up Real Earnings Growth info for Bayesian Inference
gramMatrix_Reg = (REG).dot(REG)
invMatrix_Reg = 1/gramMatrix_Reg

#Set up Cumulative Premium info for Bayesian Inference
timeVector = numpy.arange(NSTEPS-1)
heatVector = Heat[:-1]
# deviation minus trend
print('centered deviation plot')
dev = [item - IDY for item in Deviation]
pyplot.plot(dev)
pyplot.show()
# detrended heat 
print('detrended heat plot')
detrendedHeat = [Heat[i] - IDY * i for i in range(NSTEPS)]
pyplot.plot(detrendedHeat)
pyplot.show()
#Heat, T, Spread
# computation of the Gram matrix for Bayesian inference
F = Y.values
FT = F.transpose()
Gram = FT.dot(F)
invMatrix_Dev = numpy.linalg.inv(Gram)# inverting this matrix for Bayesian posterior simulation
devPointEstimate = Reg.params# point estimates for parameters of regression
devV_hat = stderr * stderr# point estimate for variance of residuals

print("------------------Simulation with Bayesian-----------------------")
#Current Value = Starting Value for year
startingTime = timeVector[-1]
#and for (not yet detrended) Heat
startingHeat = Heat[-1]
simulationResults = []
for x in range(NSIMS):
    #mean and variance posterior simulation for real earnings growth
    regBayesVar, regBayesMean = BayesSim(meanREG, stdREG*stdREG, invMatrix_Reg, numpy.size(REG)-1)
    # simulation of regression coefficients and standard error from multivariate normal-inverse chi squared posterior
    devVar, devCoeff = BayesSimMulti(devPointEstimate, devV_hat, invMatrix_Dev, NSTEPS-5)
    # we will update the heat (not detrended so far) measure
    currentHeat = startingHeat
    #initial value of deviation = implied dividend yield
    currentDeviation = 0 
    #total real return
    currentRun = 0
    # simulation of error terms = residuals
    errorTerms = numpy.random.normal(0, 1, HORIZONSIM)
    # real earnings growth terms simulation
    REGTerms = numpy.random.normal(regBayesMean, regBayesVar, HORIZONSIM)
    for j in range(HORIZONSIM):
        # computing the result of the main regression with simulated real earnings growth and residual terms
        currentDeviation = devCoeff[0] + (devCoeff[1]*REGTerms[j]) + (devCoeff[2]*currentHeat) + (devCoeff[3]*(startingTime+j))  + numpy.sqrt(regBayesVar) * errorTerms[j]
        # adding implied dividend yield to previous value of (not yet detrended) heat to get new value
        currentHeat += currentDeviation
        # adding implied dividend yield and real earnings growth to cumulative total return
        currentRun += (currentDeviation + REGTerms[j])
    # adding the average annualized real total return to the list of such 10000 returns
    simulationResults.append(currentRun/(STEP*HORIZONSIM))
print("")
print('Simulations with Current Values and ', STEP * HORIZONSIM, ' Years')
print('Expected Average Annualized Total Real Return')
print(numpy.mean(simulationResults))
print('')
print('Standard Deviation of Average Annualized Total Real Return')
print(numpy.std(simulationResults))
print("")
ordered = numpy.sort(simulationResults)
print("90% Value at Risk: ", ordered[int(0.1*NSIMS)])
print("")
print("95% Value at Risk: ", ordered[int(0.05*NSIMS)])
# Next, we do histogram of annual average total real returns
n, bins, patches = pyplot.hist(x=simulationResults, bins='auto', color='#0504aa',alpha=0.7, rwidth=1)
pyplot.grid(axis='y', alpha=0.75)
pyplot.xlabel('Average Annualized Total Real Return')
pyplot.ylabel('Frequency')
pyplot.title('Simulations with Current Values')
maxfreq = n.max()
pyplot.ylim(ymax=numpy.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
pyplot.show()       
