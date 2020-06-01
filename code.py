import pandas as pd
import numpy as np
import statsmodels
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels import api

# Random Seed for reproducibility
np.random.seed(1234)
# number of years, 1871-2019
N = 149
annualDF = pd.read_excel('Total.xlsx', sheet_name = 'Annual')
annual = annualDF.values
div = annual[:N, 1] #annual dividends
earn = annual[:N, 2] #annual earnings
monthlyDF = pd.read_excel('Total.xlsx', sheet_name = 'Monthly')
monthly = monthlyDF.values 
index = monthly[::12, 1].astype(float) #annual index values
cpi = monthly[::12, 2].astype(float) #annual consumer price index
ldiv = np.log(div) #logarithmic dividends
learn = np.log(earn) #logarithmic earnings 
lindex = np.log(index) 
lcpi = np.log(cpi)

#Exploratory plots; data is plotted versus years
def exp_plot(data, years, ylabel_text, title_text, log_flag = True):
    plt.figure(figsize=(7,6))
    plt.plot(range(years[0], years[1]), data)
    plt.xlabel('Years', fontsize = 7)
    plt.ylabel(ylabel_text, fontsize = 7)
    plt.title(title_text, fontsize = 10)
    if log_flag == True:
        plt.yscale('log')
    plt.show()

#Real values of dividends, earnings, and index    
rdiv = cpi[-1]*div/cpi[1:]
rearn = cpi[-1]*earn/cpi[1:]
rindex = cpi[-1]*index/cpi
TR = np.array([np.log(div[k] + index[k+1]) - np.log(index[k]) for k in range(N)]) #Total nominal return
lwealth = np.append([0], np.cumsum(TR)) #logarithmic nominal wealth for 1$ invested in Jan 1871, dividends reinvested
wealth = np.exp(lwealth) #nominal wealth for 1$ invested in Jan 1871, dividends reinvested
earngr = np.diff(learn) #nominal earnings growth
infl = np.diff(lcpi) #logarithmic inflation rates
realret = TR - infl #total real return
lrwealth = lwealth - lcpi + lcpi[0]*np.ones(N+1) #logarithmic real wealth, 1$ = Jan 1871
rwealth = np.exp(lrwealth) #real wealth
rearngr = earngr - infl[1:] #real earnings growth
lrearn = np.log(rearn) #logarithmic real earnings
rearngr = np.diff(lrearn) #real earnings growth
idivyield = realret[1:] - rearngr #implied dividend yield

#Now we can just call the function with all the exploratory plots
exp_plot(div, (1871,2020), 'Dividends', 'Log plot of nominal dividends', True)
exp_plot(rdiv, (1871,2020), 'Dividends', 'Log plot of real dividends', True)
exp_plot(earn, (1871,2020), 'Earnings', 'Log plot of nominal earnings', True)
exp_plot(rearn, (1871,2020), 'Earnings', 'Log plot of real earnings', True)
exp_plot(index, (1871,2021), 'Index', 'Log plot of nominal index', True)
exp_plot(rindex, (1871,2021), 'Index', 'Log plot of real index', True)
exp_plot(cpi, (1871,2021), 'CPI', 'Log plot of CPI')
exp_plot(wealth, (1871, 2021), 'Nominal wealth, Jan.1871 = $1', 'Log plot of nominal wealth', True) 
exp_plot(rwealth, (1871, 2021), 'Real wealth, Jan.1871 = $1', 'Log plot of real wealth', True)

plt.hist(rearngr, bins = 20)
plt.title('Histogram of 148 real earnings growth data')
plt.show()
qqplot(rearngr, line = 's')
plt.title('QQ plot for real earnings growth')
plt.show()

#mean and stdev for real earnings growth
meanREG = np.mean(rearngr)
stdREG = np.std(rearngr)
print('Mean of real earnings growth = ', round(meanREG, 5))
print('Std of real earnings growth = ', round(stdREG, 5))

#autocorrelation analysis of real earnings growth time series
#autocorrelation function with confidence intervals
plot_acf(rearngr, unbiased = True)
plt.show()

#p-value plot for Ljung-Box statistics
autocorrREG, statREG, pREG = acf(rearngr, unbiased = True, fft = False, qstat = True)
plt.plot(range(1, len(autocorrREG)), pREG)
plt.xlabel('Lag')
plt.ylabel('p')
plt.title('Ljung-Box p-value')
plt.show()

#Hill estimator for real earnings growth; first right tail
lREGp = sorted(np.log(rearngr[rearngr > 0])) #logarithms of positive REG terms
lREGp = lREGp[::-1] #sorted from top to bottom
lengthp = len(lREGp)
hillplus = [np.mean(lREGp[:k]) - lREGp[k] for k in range(1, lengthp)]
plt.plot(hillplus[:int((lengthp-1)*0.8)])
plt.title('Hill plot: right tail of real earnings growth')
plt.xlabel('k')
plt.ylabel('Hill value')
plt.show()

#Modified Hill plot for right tail
Theta = np.arange(0.01, 0.8, 0.01)
plt.plot(Theta, [hillplus[int((lengthp-1)**theta)] for theta in Theta])
plt.title('Modified Hill plot: right tail of real earnings growth')
plt.xlabel('theta')
plt.ylabel('Hill value')
plt.show()

#Hill estimator for real earnings growth; now left tail
lREGm = sorted(np.log(-rearngr[rearngr < 0])) #logarithms of positive REG terms
lREGm = lREGm[::-1] #sorted from top to bottom
lengthm = len(lREGm)
hillminus = [np.mean(lREGm[:k]) - lREGm[k] for k in range(1, lengthm)]
plt.plot(hillminus[:int((lengthm-1)*0.8)])
plt.title('Hill estimate for left tail of real earnings growth')
plt.xlabel('k')
plt.ylabel('Hill value')
plt.show()

#Modified Hill plot: left tail
Theta = np.arange(0.01, 0.8, 0.01)
plt.plot(Theta, [hillminus[int((lengthm-1)**theta)] for theta in Theta])
plt.title('Modified Hill plot: left tail of real earnings growth')
plt.xlabel('theta')
plt.ylabel('Hill value')
plt.show()

#mean and stdev for total real returns
Mean = np.mean(realret)
Stdev = np.std(realret)
print('Mean of total real return = ', round(Mean, 5))
print('Stdev of total real return = ', round(Stdev, 5))
# Make the plot
exp_plot(idivyield, (1871, 2019), 'Implied dividend yield', 'Implied dividend yield: total return minus earnings growth', False)
#mean and stdev for implied dividend yield
meanidy = np.mean(idivyield)
stdidy = np.std(idivyield)
print('Annualized average of implied dividend yield = ', round(meanidy, 5))
print('Annualized stdev of implied dividend yield = ', round(stdidy, 5))

#bubble not yet detrended measure: cumulative sum of implied dividend yield terms
Bubble = np.append([0], np.cumsum(idivyield))
# Time to plot
exp_plot(Bubble, (1871, 2020), 'Bubble measure', 'Cumulative implied dividend yield, not detrended', False)
#Regression factors: Design matrix
Y = pd.DataFrame({'Const': 1, 'REG': rearngr, 'Bubble': Bubble[:-1], 'Trend': range(N-1)})
#main regression fit
Reg = api.OLS(idivyield, Y).fit()
print(Reg.summary()) #Regression results
intercept = Reg.params[0] #alpha from article
REG_coeff = Reg.params[1] #gamma from article
bubble_measure_coeff = Reg.params[2] #minus beta from article
trend_coeff = Reg.params[3] #b = minus beta times c (=long-term implied dividend yield)

#Regression Residuals Analysis
Residuals = idivyield - Reg.predict(Y) #regression residuals
stderr = np.sqrt((1/(N-5))*np.dot(Residuals, Residuals)) #standard error for regression residuals
print('standard error residuals = ', round(stderr, 5)) 
qqplot(Residuals, line = 's') 
plt.title('QQ plot for residuals')
plt.show()
print('Normality testing of residuals')
print('Shapiro-Wilk p = ', stats.shapiro(Residuals)[1])
print('Jarque-Bera p = ', stats.jarque_bera(Residuals)[1])
print('Kendall Correlation between residuals and REG = ')
print(stats.kendalltau(rearngr, Residuals))
print('Spearman Correlation between residuals and REG = ')
print(stats.spearmanr(rearngr, Residuals))

#Autocorrelation analysis for regression residuals
#Durbin-Watson test is already in the general regression results
#Autocorrelation graph with confidence intervals
#This shows there is no autocorrelation
plot_acf(Residuals, unbiased = True)
plt.show()
#p-values for Ljung-Box test
autocorr_res, stat_res, p_res = acf(Residuals, unbiased = True, fft=False, qstat = True)
plt.plot(range(1, len(p_res)+1), p_res)
plt.xlabel('Lag')
plt.ylabel('p')
plt.title('Ljung-Box p-value for regression residuals')
plt.show()

IDY = trend_coeff/abs(bubble_measure_coeff) #long-term average of implied dividend yield c in article
print('Trend coefficient is ', round(IDY, 5))
detrend_bubble_measure = Bubble - IDY*range(N) #detrended bubble measure
exp_plot(detrend_bubble_measure, (1871, 2020), 'detrended bubble measure', 'Time detrended bubble measure', False)
print('Current detrended bubble measure = ', round(detrend_bubble_measure[-1], 5))
print('Avg real earnings growth = ', round(meanREG, 5)) #g from article
long_term_bubble = (IDY - intercept - REG_coeff * meanREG)/bubble_measure_coeff #long-term bubble measure, h from article
print('Long-term bubble measure = ', round(long_term_bubble, 5))
current_bubble = detrend_bubble_measure[-1] #bubble measure as of January 2020
print('Current bubble measure = ', current_bubble)
limtrret = meanREG + IDY #long-run total real returns, c + g from article
sigma2 = (REG_coeff*stdREG)**2 + stderr**2
print('Long-term total real return = ', round(limtrret, 5))
limitvar = sigma2/(1 - (1 + bubble_measure_coeff)**2)
condvar = (1 + REG_coeff)**2*(stdREG**2) + stderr**2
limitingR2 = (bubble_measure_coeff**2)*limitvar/(condvar + (bubble_measure_coeff**2)*limitvar)
print('Long-term r squared = ', limitingR2)
print('Comparison with P/E and CAPE')
print('Correlation coefficient for P/E ratio')
print(stats.pearsonr([earn[k]/index[k+1] for k in range(N-1)], realret[1:])[0])
print('Correlation coefficient for CAPE')
print(stats.pearsonr([np.mean(earn[k:k+10])/index[k+11] for k in range(N-10)], realret[10:])[0])

#Kernel density estimation for real earnings growth terms
def kernel(nsims):
    output = []
    rand = np.random.normal(0, 1, nsims)
    H = ((4/3)**0.2)*stdREG*(N**(-0.2))
    for sim in range(nsims):
        x = np.random.choice(rearngr)
        result = x + H*rand[sim]
        output.append(result)
    return output

NSIMS = 10000 #Number of simulations
regsim = kernel(NSIMS) #Simulate many i.i.d. real earnings growth terms
plt.hist(regsim, bins = 100)
plt.title('Real earnings growth density')
plt.show() #histogram of simulated real earnings growth

#one simulation: initial_bubble is the H(0) from article, no withdrawals
def simulation(horizon, initial_bubble):
    growth = kernel(horizon)
    error = np.random.normal(0, 1, horizon)
    idys = [] #here we write our implied dividend yield terms
    #this is cumulative sum of implied dividend yield terms: Bubble measure, not detrended
    Current_bubble = initial_bubble 
    for t in range(horizon):
        idy = intercept + REG_coeff * growth[t] + bubble_measure_coeff * Current_bubble + trend_coeff * t + stderr * error[t]
        Current_bubble = Current_bubble + idy
        idys.append(idy)
    avgidy = np.mean(idys) #average simulated implied dividend yield terms
    avgREG = np.mean(growth) #average simulated real earnings growth terms
    return avgidy + avgREG #average simulated total real return

#many simulations, initial bubble is the H(0) from article, no withdrawals
def finalSim(horizon, initial_bubble):
    print('Horizon = ', horizon)
    print('Bubble = ', initial_bubble)
    output = []
    caption = str(horizon) + ' years, initial bubble measure ' + str(round(initial_bubble, 3))
    for sim in range(NSIMS):
        output.append(simulation(horizon, initial_bubble))
    plt.hist(output, bins = 100)
    plt.xlabel('average total real return')
    plt.title(caption)
    plt.show()
    print('VaR 99% = ', round(np.percentile(output, 1), 5))
    print('VaR 95% = ', round(np.percentile(output, 5), 5))
    print('VaR 90% = ', round(np.percentile(output, 10), 5))
    print('mean = ', round(np.mean(output), 5))
    print('stdev = ', round(np.std(output), 5))
    return 0

print('Current market conditions')
finalSim(5, current_bubble) #5 years
finalSim(10, current_bubble) #10 years
finalSim(20, current_bubble) #20 years

#one simulation, initial_bubble H(0) from article, withdraw_rate is the
#withdrawal rate fraction of initial wealth, inflation-adjusted
def simulation_withdraw(horizon, withdraw_rate, initial_bubble):
    growth = kernel(horizon)
    error = np.random.normal(0, 1, horizon)
    Current_bubble = initial_bubble
    Current_wealth = 1
    t = 0
    boolean = True #Break the loop after horizon steps or after ruin
    while boolean: 
        idy = intercept + REG_coeff * growth[t] + bubble_measure_coeff * Current_bubble + trend_coeff * t + stderr * error[t]
        Current_bubble = Current_bubble + idy
        Current_wealth = (Current_wealth - withdraw_rate) * np.exp(idy + growth[t])
        t = t + 1
        if (Current_wealth < 0): #Ruin 
            boolean = False
        if (t == horizon): #End of simulation, reached time horizon without ruin
            boolean = False                
    if (Current_wealth > 0): #The case of no ruin, reached time horizon
        return ('fine', np.log(Current_wealth))
    if (Current_wealth < 0): #The case of ruin
        return ('ruin', t)

#many simulations, given withdrawal rate and initial bubble measure
def withdrawal(horizon, withdraw_rate, initial_bubble):
    print('Horizon = ', horizon)
    print('Bubble = ', initial_bubble)
    print('Withdrawal rate = ', withdraw_rate)
    caption = 'horizon ' + str(horizon) + ' withdraw ' + str(withdraw_rate) + ' initial bubble measure ' + str(round(initial_bubble, 4))
    all_wealth = [] #Here we write log wealth for sims when no ruin
    all_ruined = [] #Here we write ruin time for sims with ruin
    for sim in range(NSIMS):
        result, output = simulation_withdraw(horizon, withdraw_rate, initial_bubble)
        if result == 'ruin':
            all_ruined.append(output)
        if result == 'fine':
            all_wealth.append(output)
    print('Ruin probability', len(all_ruined)/NSIMS)
    if (len(all_wealth) > 0): #If there are some sims with no ruin
        plt.hist(all_wealth, bins = 100)
        plt.xlabel('Log terminal wealth')
        plt.title(caption)
        plt.show()
        #These values at risk, mean, and stdev are conditional upon no ruin
        print('VaR 99% log terminal wealth = ', round(np.percentile(all_wealth, 1), 5))
        print('VaR 95% log terminal wealth = ', round(np.percentile(all_wealth, 5), 5))
        print('VaR 90% log terminal wealth = ', round(np.percentile(all_wealth, 10), 5))
        print('mean log terminal wealth = ', round(np.mean(all_wealth), 5))
        print('stdev log terminal wealth = ', round(np.std(all_wealth), 5))
    if (len(all_ruined) > int(0.01*NSIMS)): #If ruin probability is > 1%
        plt.hist(all_ruined)
        plt.title('Ruin time')
        plt.show()
        print('mean ruin time = ', round(np.mean(all_ruined), 5)) #mean of ruin time
        print('stdev ruin time = ', round(np.std(all_ruined), 5)) #stdev of ruin time
    #breakeven probability: even after withdrawals, terminal wealth is greater than initial wealth 1
    return sum([item > 0 for item in all_wealth])/NSIMS 

#simulations for 5, 10, 20 years
withdrawal_rate = 0.04 #classic withdrawal rate
print('Withdrawal rates', withdrawal_rate)
print('Long term bubble measure') #starting from H(0) = h
print(withdrawal(5, withdrawal_rate, long_term_bubble))
print(withdrawal(10, withdrawal_rate, long_term_bubble))
print(withdrawal(20, withdrawal_rate, long_term_bubble))
print('Current bubble measure') #starting from H(0) = current_bubble = -0.0468
print(withdrawal(5, withdrawal_rate, current_bubble))
print(withdrawal(10, withdrawal_rate, current_bubble))
print(withdrawal(20, withdrawal_rate, current_bubble))





