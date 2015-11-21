import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

loansData['Interest.Rate'] = pd.to_numeric(loansData['Interest.Rate'].apply(lambda x : round(float(x.replace('%', ''))/100,4) ))

loansData['Loan.Length'] = pd.to_numeric(loansData['Loan.Length'].apply(lambda x : x.replace(' months', '')))

loansData['FICO.Score'] = pd.to_numeric(loansData['FICO.Range'].apply(lambda x : x.split('-')[0]))

##plt.figure()
##p = loansData['FICO.Score'].hist()
##plt.show()

##a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(15,15), diagonal='hist')

intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

# The dependent variable
y = np.matrix(intrate).transpose()
# The independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

x = np.column_stack([x1,x2])

X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

f.summary()
