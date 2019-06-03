from flask import Flask, render_template, request, redirect, url_for
import tablib
from random import choice
from werkzeug.utils import secure_filename
import base64


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as rcParams
import io
import os, fnmatch
import seaborn as sns
import psycopg2

# get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

# ### Importing Required csv files

# rcParams['figure.figsize'] = 20,10
data_time = pd.read_csv('Hourly.csv')
data_time.head(10)

# In[4]:


data_time.shape

# In[5]:


data_monthly = pd.read_csv('Monthly.csv')

# In[6]:
def monthly():
  data12=data_monthly
  return data12

data_monthly.head(10)

# In[7]:


data_monthly.shape

# ### See the unique States present in dataframe

# In[8]:


state_names = data_monthly['STATE/UT'].unique()
state_names

# In[9]:


data_monthly['STATE/UT'] = data_monthly['STATE/UT'].replace({'D & N Haveli': 'D&N Haveli', 'Delhi (Ut)': 'Delhi Ut'})

# In[10]:


data_monthly['STATE/UT'].unique()

# ### Creating SUMMER, AUTUMN, WINTER, SPRING columns in data_monthly and droping the Months column

# In[11]:


data_monthly['SUMMER'] = data_monthly[['JUNE', 'JULY', 'AUGUST']].sum(axis=1)
data_monthly['AUTUMN'] = data_monthly[['SEPTEMBER', 'OCTOBER', 'NOVEMBER']].sum(axis=1)
data_monthly['WINTER'] = data_monthly[['DECEMBER', 'JANUARY', 'FEBRUARY']].sum(axis=1)
data_monthly['SPRING'] = data_monthly[['MARCH', 'APRIL', 'MAY']].sum(axis=1)

# Now you can drop the columns from Jan to Dec

data_monthly.drop(columns=['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER',
                           'OCTOBER', 'NOVEMBER', 'DECEMBER'], axis=1, inplace=True)

# In[12]:


# Now group the state columns by summing the values of season
state_grouped = data_monthly.groupby(['STATE/UT']).sum()

# In[13]:


state_grouped.head()

# ### Now calculating the % of every state met with an accident in each Season

# In[14]:


# Now do the percentage of every state met with an accident in each season
def graph_season_acci():
  state_grouped['%_SUMMER'] = state_grouped['SUMMER'] / state_grouped['TOTAL']
  state_grouped['%_AUTUMN'] = state_grouped['AUTUMN'] / state_grouped['TOTAL']
  state_grouped['%_WINTER'] = state_grouped['WINTER'] / state_grouped['TOTAL']
  state_grouped['%_SPRING'] = state_grouped['SPRING'] / state_grouped['TOTAL']
  result1 = state_grouped.iloc[:, 1:]
  return result1

# ### Now Let's see the accident happend at  Time Intervals

# In[15]:


cols = ['STATE/UT', 'YEAR', '0-3', '3-6', '6-9', '9-12', '12-15', '15-18', '18-21', '21-24', 'Total']
data_time.columns = cols

# In[16]:


data_time.head(10)

# ### Now doing the sum of all accidents happend at different intervals in Each State

data_time_grouped = data_time.groupby(['STATE/UT']).sum()

data_time_grouped.head()

# ### Now calculating the % of accidents occured at different Timings

# Now group the different data timings to category like Morning, AfterNoon, Evening, Night
data_time_grouped['_%NIGHT'] = (data_time_grouped['0-3'] + data_time_grouped['3-6']) / data_time_grouped['Total']
data_time_grouped['_%MORNING'] = (data_time_grouped['6-9'] + data_time_grouped['9-12']) / data_time_grouped['Total']
data_time_grouped['_%AFTER_NOON'] = (data_time_grouped['12-15'] + data_time_grouped['15-18']) / data_time_grouped[
    'Total']
data_time_grouped['_%EVENING'] = (data_time_grouped['18-21'] + data_time_grouped['21-24']) / data_time_grouped['Total']

data_time_grouped.head()

# ### Now droping the columns of different timings [0-3, 3-6,....21-24]

# Now drop the columns of 0-3,3-6,......,21-24
def time_acci():
  data_time_grouped.drop(data_time_grouped.columns[0:9], axis=1, inplace=True)
  result2 = data_time_grouped
  return result2

# ### Now plot the graph of more accidents happend at different seasons

# Now plot the graph of data_time_grouped in the form of pie chart so that we can see in which season more 
# accidents has met
def season_plot():
  state_grouped.loc[:, 'SUMMER':'SPRING'].sum(axis=0).plot.pie(
    title='Seasonal Distribution of all accidents in 2001-2014', autopct='%1.1f%%')
  plt.savefig('static/seasonal_acci_plot.png')

# ### Now Let's see the state that has met highest percentage of accidents
# Lets find out the state with highest percentage across different seasons

# def highest_state_acci():
#   # States with highest % of accidents in different time blocks
#   plt.figure(figsize = (18,4))
#   state_grouped.sort_values('%_SUMMER',ascending = False).head().loc[:,['STATE/UT','%_SUMMER']].plot(kind = 'bar', ax = plt.subplot(141),color = 'b', title = 'Higher Summer Accidents')

#   state_grouped.sort_values('%_WINTER', ascending = False).head().loc[:,['STATE/UT','%_WINTER']].plot(kind = 'bar',ax = plt.subplot(142),color = 'r', title = 'Higher Winter Accidents')

#   state_grouped.sort_values('%_AUTUMN', ascending = False).head().loc[:,['STATE/UT','%_AUTUMN']].plot(kind = 'bar',ax = plt.subplot(143),color = 'g', title = 'Higher Autumn Accidents') 

#   state_grouped.sort_values('%_SPRING',ascending = False).head().loc[:,['STATE/UT','%_SPRING']].plot(kind = 'bar',ax = plt.subplot(144), color = 'y', title = 'Higher Autumn Accidents')

#   plt.savefig('static/highest_state_acci.png')

# # ### Let's see the Year wise, which state has met with more number of Accidents
# # Now Let's see Year wise higher accidents of states

def year_wise_acci():
  highest_accident_states = state_grouped.sort_values('TOTAL', ascending=False)
  high_states = list(highest_accident_states.head().index)
  df4 = data_monthly.loc[data_monthly['STATE/UT'].isin(high_states), ['STATE/UT', 'YEAR', 'TOTAL']]

  plt.figure(figsize=(10, 6))
  ax = plt.subplot(111)
  for key, grp in df4.groupby(['STATE/UT']):
      ax = ax = grp.plot(ax=ax, kind='line', x='YEAR', y='TOTAL', label=key)
  # plt.show()

  plt.savefig('static/year_wise_acci.png')

# # ### List of top five states met with more number of Accidents

# # In[26]:


# # List of five states met with high accidents
# states_list = print(high_states)

# # ## Now Let's see the accidents met over a day
# ## Break up accidents for all states over the time blocks:
# # state_time_grouped.info()
def daily_acci():
  df2 = data_time_grouped.sum(axis=0)
  df2.drop(['Total']).T.plot.pie(title='All accidents 2001-2014', subplots=True, figsize=(5, 5), autopct='%1.1f%%')
  df2 = data_time_grouped.sum(axis=0)

  plt.savefig('static/daily_acci.png')

# # ### Now Let's see how Accidents has grown over years
# # Let's see how accidents has grown over years
# # Notice that accidents are increased exponentially with the years after 2001

# def acci_over_years():
df3 = data_time.groupby(['YEAR']).sum()
df3.loc[:, 'Total'].plot(kind='line', title='Incresed Accidents over Years')  

  # plt.savefig('static/acci_over_years.png')


# # States with highest % of accidents in different time blocks
def acci_diff_times():
  plt.figure(figsize=(18, 4))
  data_time_grouped.sort_values('_%MORNING', ascending=False).head().loc[:, ['STATE/UT', '_%MORNING']].plot(kind='bar',
                                                                                                          ax=plt.subplot(
                                                                                                              141),
                                                                                                          color='b')

  data_time_grouped.sort_values('_%AFTER_NOON', ascending=False).head().loc[:, ['STATE/UT', '_%AFTER_NOON']].plot(
    kind='bar', ax=plt.subplot(142), color='r')

  data_time_grouped.sort_values('_%EVENING', ascending=False).head().loc[:, ['STATE/UT', '_%EVENING']].plot(kind='bar',
                                                                                                          ax=plt.subplot(
                                                                                                              143),
                                                                                                          color='g')

  data_time_grouped.sort_values('_%NIGHT', ascending=False).head().loc[:, ['STATE/UT', '_%NIGHT']].plot(kind='bar',
                                                                                                      ax=plt.subplot(
                                                                                                          144),
                                                                                                      color='y')
  plt.savefig('static/acci_diff_times.png')

# # ## Now Let's see the Accidents met w.r.t Ages

ages = pd.read_csv('Ages.csv', sep='\t')
ages.head()

ages.drop(axis=1, columns=['Age 0-14 - 2014', 'Age 15-24 - 2014', 'Age 25-64 - 2014', 'Age 65 & Above - 2014'],
          inplace=True)

ages.drop(axis=1, columns='S. No.', inplace=True)

ages.head()

cols = ['State/UT', 'Fatal_18-', 'Total_18-', 'Fatal_18-25', 'Total_18-25', 'Fatal_25-35', 'Total_25-35', 'Fatal_35-45',
        'Total_35-45', 'Fatal_45-60', 'Total_45-60', 'Fatal_60+', 'Total_60+', 'No-Age-Fatal', 'No-Age-Total']

ages.columns = cols

ages.set_index('State/UT', inplace=True)

ages.shape

ages.tail()

ages.drop(index='Total', axis=0, inplace=True)

def ages_acci():
  age = ages
  return age

def ages_graph1():
  fig = plt.figure(figsize=(15, 3))
  ages.sort_values('Fatal_18-', ascending=False).head().loc[:, ['State/UT', 'Fatal_18-']].plot(kind='bar',
                                                                                             ax=fig.add_subplot(141),
                                                                                             color='r')

  ages.sort_values('Total_18-', ascending=False).head().loc[:, ['State/UT', 'Total_18-']].plot(kind='bar',
                                                                                             ax=fig.add_subplot(142),
                                                                                             color='g')

  ages.sort_values('Fatal_18-25', ascending=False).head().loc[:, ['State/UT', 'Fatal_18-25']].plot(kind='bar',
                                                                                                 ax=fig.add_subplot(
                                                                                                     143), color='b')

  ages.sort_values('Total_18-25', ascending=False).head().loc[:, ['State/UT', 'Total_18-25']].plot(kind='bar',
                                                                                                 ax=fig.add_subplot(
                                                                                                     144), color='y')
  plt.savefig('static/ages_graph1.png')

def ages_graph2():
  fig = plt.figure(figsize=(15, 6))
  ages.sort_values('Fatal_25-35', ascending=False).head().loc[:, ['State/UT', 'Fatal_25-35']].plot(kind='bar',
                                                                                                 ax=fig.add_subplot(
                                                                                                     241), color='y')

  ages.sort_values('Total_25-35', ascending=False).head().loc[:, ['State/UT', 'Total_25-35']].plot(kind='bar',
                                                                                                 ax=fig.add_subplot(
                                                                                                     242), color='b')

  ages.sort_values('Fatal_35-45', ascending=False).head().loc[:, ['State/UT', 'Fatal_35-45']].plot(kind='bar',
                                                                                                 ax=fig.add_subplot(
                                                                                                     243), color='g')

  ages.sort_values('Total_35-45', ascending=False).head().loc[:, ['State/UT', 'Total_35-45']].plot(kind='bar',
                                                                                                 ax=fig.add_subplot(
                                                                                                     244), color='r')
  plt.savefig('static/ages_graph2.png')
# # In[16]:

def ages_graph3():
  fig = plt.figure(figsize=(15, 8))
  ages.sort_values('Fatal_45-60', ascending=False).head().loc[:, ['State/UT', 'Fatal_45-60']].plot(kind='bar',
                                                                                                 ax=fig.add_subplot(
                                                                                                     341), color='y')

  ages.sort_values('Total_45-60', ascending=False).head().loc[:, ['State/UT', 'Total_45-60']].plot(kind='bar',
                                                                                                 ax=fig.add_subplot(
                                                                                                     342), color='b')

  ages.sort_values('Fatal_60+', ascending=False).head().loc[:, ['State/UT', 'Fatal_60+']].plot(kind='bar',
                                                                                             ax=fig.add_subplot(343),
                                                                                             color='g')

  ages.sort_values('Total_60+', ascending=False).head().loc[:, ['State/UT', 'Total_60+']].plot(kind='bar',
                                                                                             ax=fig.add_subplot(344),
                                                                                             color='r')
  plt.savefig('static/ages_graph3.png')

ages1 = pd.DataFrame(data=ages.index)

ages1.set_index('State/UT', inplace=True)

ages1['%_Age_18-'] = ages['Fatal_18-'] / ages['Total_18-']

ages1['%_Age_18-25'] = ages['Fatal_18-25'] / ages['Total_18-25']

ages1['%_Age_25-35'] = ages['Fatal_25-35'] / ages['Total_25-35']

ages1['%_Age_35-45'] = ages['Fatal_35-45'] / ages['Total_35-45']

ages1['%_Age_45-60'] = ages['Fatal_45-60'] / ages['Total_45-60']

ages1['%_Age_60+'] = ages['Fatal_60+'] / ages['Total_60+']

ages1['%_Age_Unknown'] = ages['No-Age-Fatal'] / ages['No-Age-Total']

ages1.head()

ages1.isna().sum()

mean_18 = ages1['%_Age_18-'].mean()
mean_25 = ages1['%_Age_18-25'].mean()
mean_45 = ages1['%_Age_35-45'].mean()
mean_60 = ages1['%_Age_45-60'].mean()
mean_60_plus = ages1['%_Age_60+'].mean()
mean_unknown = ages1['%_Age_Unknown'].mean()

ages1['%_Age_18-'] = ages1['%_Age_18-'].fillna(mean_18)
ages1['%_Age_18-25'] = ages1['%_Age_18-25'].fillna(mean_25)
ages1['%_Age_35-45'] = ages1['%_Age_35-45'].fillna(mean_45)
ages1['%_Age_45-60'] = ages1['%_Age_45-60'].fillna(mean_60)
ages1['%_Age_60+'] = ages1['%_Age_60+'].fillna(mean_60_plus)
ages1['%_Age_Unknown'] = ages1['%_Age_Unknown'].fillna(mean_unknown)

ages1.isna().sum()

def final_ages_acci():
  plt.figure(figsize=(10, 8))
  ages1.loc[:, '%_Age_18-':'%_Age_Unknown'].sum(axis=0).plot.pie(title='% of Accidents in Different Ages',
                                                               autopct='%1.1f%%')

  plt.savefig('static/final_ages_acci.png')

dataset1 = tablib.Dataset()
with open(os.path.join(os.path.dirname(__file__),'final_Seasonal_csv.csv')) as f:
  dataset1.csv = f.read()

dataset2 = tablib.Dataset()
with open(os.path.join(os.path.dirname(__file__),'final_Daily_csv.csv')) as f:
  dataset2.csv = f.read()

dataset3 = tablib.Dataset()
with open(os.path.join(os.path.dirname(__file__),'final_Ages_csv.csv')) as f:
  dataset3.csv = f.read()


app = Flask(__name__)
con = psycopg2.connect("host='localhost' dbname='accesion' user='somesh' password='somesh18'")
cur = con.cursor()
@app.route('/')
def index():
  return render_template('index.html')


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        un=str(request.form['username'])
        pd=str(request.form['password'])
        if un=='admin' and pd == 'password':
            return render_template('hello.html')
        return render_template('login.html')
        
    else:
        return render_template('login.html')


@app.route('/plots', methods = ['GET','POST'])
def plots():
  if request.method == 'POST':
    option = str(request.form.get('Road_Accident'))
    if option == 'Season':
      season_plot()
      
      return render_template('state.html')

    elif option == 'Time':
      # daily_acci = 'static/daily_acci.png'
      return render_template('time.html')

    elif option == 'Age':
      final_ages_acci()
      return render_template('ages.html')

    elif option == 'Year_Wise_State_Acci':
      return render_template('year_wise_state_acci.html')

    else:
      return render_template('state_wise_ages_acci.html')
    
  else:

    return render_template('hello.html')

@app.route('/agesdata')
def agesdata():
 
  return render_template('data.html', ss1=ages.to_html(), ss2=dataset3.html, data="ages wise dataset", data1="ages wise trained dataset")

@app.route('/hourdata')
def hourdata():
 
  return render_template('data.html', ss1=data_time.to_html(), ss2=dataset2.html, data="time wise dataset", data1="time wise trained dataset")

@app.route('/monthdata')
def monthdata():
 
  return render_template('data.html', ss1=data_monthly.to_html(), ss2=dataset1.html, data="seasonal wise dataset", data1="seasonal wise trained dataset")

@app.route('/logout')
def logout():
 
  return render_template('login.html')

@app.route('/logout1')
def logout1():
 
  return render_template('login1.html')


@app.route('/login1', methods=['GET','POST'])
def login1():
    if request.method=='POST':
        un=str(request.form['username'])
        pd=str(request.form['password'])

        cur.execute("SELECT user_name, password FROM housereg where user_name = %s;",(un,))
        records = cur.fetchall()
        for i in records:
            if un==i[0] and pd == i[1]:
                return redirect('client')
        return redirect('login1')
        
    else:
        return render_template('login1.html')

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


@app.route('/client', methods = ['GET','POST'])
def client():
  if request.method == 'POST':
    option = str(request.form.get('Road_Accident'))
    if option == 'Season':
      season_plot()
      
      return render_template('cstate.html')

    elif option == 'Time':
      # daily_acci = 'static/daily_acci.png'
      return render_template('ctime.html')

    elif option == 'Age':
      final_ages_acci()
      return render_template('cage.html')

    elif option == 'Year_Wise_State_Acci':
      return render_template('cyear_wise_state_acci.html')

    else:
      return render_template('cstate_wise_ages_acci.html')
    
  else:

    return render_template('client.html')

if __name__ == "__main__":
  app.run(debug = True)
