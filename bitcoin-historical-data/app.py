from flask import Flask, render_template, request, url_for, redirect, session
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
import psycopg2
from itertools import product
from datetime import datetime
warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')


app = Flask(__name__)


df = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv')

df.head()

# Unix-time to 
df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')

# Resampling to daily frequency
df.index = df.Timestamp
df = df.resample('D').mean()

# Resampling to monthly frequency
df_month = df.resample('M').mean()

# Resampling to annual frequency
df_year = df.resample('A-DEC').mean()

# Resampling to quarterly frequency
df_Q = df.resample('Q-DEC').mean()

# PLOTS
fig = plt.figure(figsize=[15, 7])

plt.suptitle('Bitcoin exchanges, mean USD', fontsize=22)

plt.subplot(221)
plt.plot(df.Weighted_Price, '-', label='By Days')
plt.legend()

plt.subplot(222)
plt.plot(df_month.Weighted_Price, '-', label='By Months')
plt.legend()

plt.subplot(223)
plt.plot(df_Q.Weighted_Price, '-', label='By Quarters')
plt.legend()

plt.subplot(224)
plt.plot(df_year.Weighted_Price, '-', label='By Years')
plt.legend()
plt.savefig("static/img1.png", dpi = 100)
# plt.tight_layout()
#plt.show()


plt.figure(figsize=[15,7])
sm.tsa.seasonal_decompose(df_month.Weighted_Price).plot()
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])
plt.savefig("static/img6.png", dpi = 100)

#plt.show()

# Box-Cox Transformations
df_month['Weighted_Price_box'], lmbda = stats.boxcox(df_month.Weighted_Price)
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])

# Seasonal differentiation
df_month['prices_box_diff'] = df_month.Weighted_Price_box - df_month.Weighted_Price_box.shift(12)
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.prices_box_diff[12:])[1])

# Regular differentiation
df_month['prices_box_diff2'] = df_month.prices_box_diff - df_month.prices_box_diff.shift(1)
plt.figure(figsize=(15,7))

# STL-decomposition
sm.tsa.seasonal_decompose(df_month.prices_box_diff2[13:]).plot()   
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.prices_box_diff2[13:])[1])
plt.savefig("static/img2.png", dpi = 100)
#plt.show()


# Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots
plt.figure(figsize=(15,7))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(df_month.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(df_month.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax)
plt.tight_layout()
plt.savefig("static/img3.png", dpi = 100)
#plt.show()



# Initial approximation of parameters
Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D=1
d=1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model=sm.tsa.statespace.SARIMAX(df_month.Weighted_Price_box, order=(param[0], d, param[1]), 
                                        seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
    except ValueError:
        print('wrong parameters:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])

# Best Models
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())
print(best_model.summary())


# STL-decomposition
plt.figure(figsize=(15,7))
plt.subplot(211)
best_model.resid[13:].plot()
plt.ylabel(u'Residuals')
ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)

print("Dickey–Fuller test:: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])

plt.tight_layout()
plt.savefig("static/img4.png", dpi = 100)
#plt.show()

# Inverse Box-Cox Transformation Function
def invboxcox(y,lmbda):
   if lmbda == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(lmbda*y+1)/lmbda))

df_month2 = df_month[['Weighted_Price']]
date_list = [datetime(2019, 4, 30), datetime(2019, 5, 31), datetime(2019, 6, 30), datetime(2019, 7, 31), 
             datetime(2019, 8, 31), datetime(2019, 9, 30), datetime(2019, 10, 31), datetime(2019, 11, 30),
             datetime(2019, 12, 31)
             ]
df_month



future = pd.DataFrame(index=date_list, columns= df_month.columns)
df_month2 = pd.concat([df_month, future])
df_month3=df_month2

# Prediction

future = pd.DataFrame(index=date_list, columns= df_month.columns)
df_month2 = pd.concat([df_month2, future])
df_month2['forecast'] = invboxcox(best_model.predict(start=0, end=110), lmbda)
pre=pd.DataFrame(df_month2['forecast'])
pre1=pre.tail(9)
plt.figure(figsize=(15,7))
df_month2.Weighted_Price.plot()

df_month2.forecast.plot(color='r', ls='--', label='Predicted Weighted_Price')
plt.legend()
plt.title('Bitcoin exchanges, by months')
plt.ylabel('mean USD')
plt.savefig("static/img5.png", dpi = 100)
#plt.show()
con = psycopg2.connect("host='localhost' dbname='accesion' user='somesh' password='somesh18'")
cur = con.cursor()

@app.route('/admin', methods=['GET','POST'])
def admin():
    if request.method=='POST':
        un=str(request.form['username'])
        pd=str(request.form['password'])
        if un=='admin' and pd == 'password':
            return render_template('admin.html')
        return render_template('index.html')
        
    else:
        return render_template('index.html')

@app.route('/data')
def data():
	return render_template('data.html', ss=df.to_html())


@app.route('/traindata')
def traindata():
	return render_template('traindata.html', ss=df_month.to_html())


@app.route('/testdata')
def testdata():
	return render_template('testdata.html', ss=df_month3.to_html())

@app.route('/dataanalysis')
def dataanalysis():
	return render_template('dataanalysis.html')

@app.route('/prediction')
def prediction():
	return render_template('prediction.html', ss=pre.to_html())

@app.route('/', methods=['GET','POST'])
def login():
    if request.method=='POST':
        un=str(request.form['username'])
        pd=str(request.form['password'])

        cur.execute("SELECT user_name, password FROM housereg where user_name = %s;",(un,))
        records = cur.fetchall()
        for i in records:
            if un==i[0] and pd == i[1]:
                return render_template('home.html', ss=pre1.to_html())
        return render_template('login.html')
        
    else:
        return render_template('login.html')

@app.route('/reg', methods=['GET','POST'])
def reg():
    if request.method=='POST':
        un=str(request.form['username'])
        pd=str(request.form['password'])
        Name=str(request.form['name'])
        #cur.execute("CREATE TABLE housereg(uname varchar(20),user_name varchar(20),password varchar(20))")
        query =  "INSERT INTO housereg(uname,user_name,password) VALUES (%s, %s, %s);"
        data = (Name,un,pd)
        cur.execute(query, data)
        con.commit()
        return redirect('login')
        
    else:
        return render_template('reg.html')


@app.route('/logout')
def logout():
    return render_template('index.html')

@app.route('/logout1')
def logout1():
    return render_template('login.html')

if __name__ == "__main__":
	app.run(debug=True)