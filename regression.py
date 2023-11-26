
# Multiple Regression
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy import stats

dF = pd.read_excel('/Users/bahu/Kurs/Machine Learning/veri.xlsx')  # Dosyadan verileri oku
dF.size             # size of the data
dF.shape            # dimension of the data 
dF.keys()           # name of the columns of the data
dF.info()           # give general info about the data
dF.isnull().any()   # check any NAN value in the data
dF.isnull().sum()
dF.head(20)         # show the first 20 rows
# Correlation
#a = dF.corr()

X = dF.iloc[:91, 7:11].values
y = dF.iloc[:91, 11].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123, shuffle=1)
regresyon = linear_model.LinearRegression()
regresyon.fit(X_train, y_train)

# y=ax1+bx2+c ise
print('y= ',regresyon.intercept_, ' + ', regresyon.coef_[0],'x1 + ',
      regresyon.coef_[1],'x2 + ',regresyon.coef_[2],'x3 + ',
      regresyon.coef_[3],'x4')

plt.scatter(X[:,3], y, color = "m", marker = "o", s = 30)
# predicted response vector
y_pred = regresyon.intercept_ + regresyon.coef_[3]*X[:,3]
# plotting the regression line
plt.plot(X[:,3], y_pred, color = "g")

r_sq = regresyon.score(X_train, y_train)
print('Coefficient of Determination:', r_sq)
print('')
print('TAHMİNLER')
prediction = regresyon.predict(X_test)
fiyatOrt = 1110626.37
fiyatSTD = 383887.9086
newPred = (prediction * fiyatSTD) + fiyatOrt 
print(prediction)
print(newPred)

from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(prediction, y_test) # mean squared errors
from math import sqrt
RMSE = sqrt(mean_squared_error(prediction, y_test)) # Root Mean Squared Error, RMSE
print('')
print('MSE= ',MSE)
print('RMSE= ',RMSE)

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

# Standardize the predictor variables
scaler = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test = scaler.transform(X_test)

regresyon.fit(Xs_train, y_train)
prediction = regresyon.predict(Xs_test)
RMSE = np.sqrt(mean_squared_error(y_test, prediction))

# Fit a Ridge regression model
ridge = Ridge(alpha=10)
ridge.fit(Xs_train, y_train)
ridge_pred = ridge.predict(Xs_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))

# Fit a Lasso regression model
lasso = Lasso(alpha=0.01)
lasso.fit(Xs_train, y_train)
lasso_pred = lasso.predict(Xs_test)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))

print("Ridge RMSE: {:.4f}".format(ridge_rmse))
print("Lasso RMSE: {:.4f}".format(lasso_rmse))
print("Regression RMSE: {:.4f}".format(RMSE))