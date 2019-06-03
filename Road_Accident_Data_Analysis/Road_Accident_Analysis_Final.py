
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ### Importing Required csv files

data_time = pd.read_csv('Hourly.csv')
data_time.head(10)
data_time.shape

# Reading Monthly data
data_monthly = pd.read_csv('Monthly.csv')
data_monthly.head(10)
data_monthly.shape


# ### See the unique States present in dataframe
state_names = data_monthly['STATE/UT'].unique()
state_names

data_monthly['STATE/UT'] = data_monthly['STATE/UT'].replace({'D & N Haveli':'D&N Haveli','Delhi (Ut)': 'Delhi Ut'})

data_monthly['STATE/UT'].unique()


# ### Creating SUMMER, AUTUMN, WINTER, SPRING columns in data_monthly and droping the Months column

data_monthly['SUMMER'] = data_monthly[['JUNE','JULY','AUGUST']].sum(axis=1)
data_monthly['AUTUMN'] = data_monthly[['SEPTEMBER','OCTOBER','NOVEMBER']].sum(axis=1)
data_monthly['WINTER'] = data_monthly[['DECEMBER','JANUARY','FEBRUARY']].sum(axis=1)
data_monthly['SPRING'] = data_monthly[['MARCH','APRIL','MAY']].sum(axis=1)

# Now you can drop the columns from Jan to Dec

data_monthly.drop(columns = ['JANUARY','FEBRUARY','MARCH','APRIL','MAY','JUNE','JULY','AUGUST','SEPTEMBER',
                             'OCTOBER','NOVEMBER','DECEMBER'],axis=1,inplace = True)

# Now group the state columns by summing the values of season
state_grouped = data_monthly.groupby(['STATE/UT']).sum()

state_grouped.head()


# ### Now calculating the % of every state met with an accident in each Season

# Now do the percentage of every state met with an accident in each season
state_grouped['%_SUMMER']=state_grouped['SUMMER']/state_grouped['TOTAL']
state_grouped['%_AUTUMN']=state_grouped['AUTUMN']/state_grouped['TOTAL']
state_grouped['%_WINTER']=state_grouped['WINTER']/state_grouped['TOTAL']
state_grouped['%_SPRING']=state_grouped['SPRING']/state_grouped['TOTAL']

state_grouped.iloc[:,1:].head()


# ### Now Let's see the accident happend at  Time Intervals 

cols = ['STATE/UT', 'YEAR', '0-3','3-6', '6-9', '9-12','12-15','15-18','18-21','21-24','Total']
data_time.columns = cols

data_time.head(10)


# ### Now doing the sum of all accidents happend at different intervals in Each State

data_time_grouped = data_time.groupby(['STATE/UT']).sum()

data_time_grouped.head()


# ### Now calculating the % of accidents occured at different Timings

# In[19]:


# Now group the different data timings to category like Morning, AfterNoon, Evening, Night
data_time_grouped['_%NIGHT'] = (data_time_grouped['0-3'] + data_time_grouped['3-6'])/data_time_grouped['Total']
data_time_grouped['_%MORNING'] = (data_time_grouped['6-9'] + data_time_grouped['9-12'])/data_time_grouped['Total']
data_time_grouped['_%AFTER_NOON'] = (data_time_grouped['12-15'] + data_time_grouped['15-18'])/data_time_grouped['Total']
data_time_grouped['_%EVENING'] = (data_time_grouped['18-21'] + data_time_grouped['21-24'])/data_time_grouped['Total']


# In[20]:


data_time_grouped.head()


# ### Now droping the columns of different timings [0-3, 3-6,....21-24]

# In[21]:


#Now drop the columns of 0-3,3-6,......,21-24
data_time_grouped.drop(data_time_grouped.columns[0:9], axis = 1,inplace = True)

data_time_grouped.head()


# ### Now plot the graph of more accidents happend at different seasons  

# Now plot the graph of data_time_grouped in the form of pie chart so that we can see in which season more 
# accidents has met
def graph1():
    state_grouped.loc[:,'SUMMER':'SPRING'].sum(axis=0).plot.pie(title = 'Seasonal Distribution of all accidents in 2001-2014',autopct='%1.1f%%')
    plt.savefig('static/graph1.png')

# ### Now Let's see the state that has met highest percentage of accidents

# Lets find out the state with highest percentage across different seasons

def graph2():
    plt.figure(figsize = (20,5))
    plt.subplot(141)
    summer = state_grouped.sort_values('%_SUMMER')
    summer['%_SUMMER'].tail(5).plot.bar(title = 'Higher Summer Accidents')
    plt.savefig('static/High_Sum_Acc.png')

    plt.subplot(142)
    winter = state_grouped.sort_values('%_WINTER')
    winter['%_WINTER'].tail(5).plot.bar(title = 'Higher Winter Accidents')
    plt.savefig('static/High_Win_Acc.png')

    plt.subplot(143)
    autumn = state_grouped.sort_values('%_AUTUMN')
    autumn['%_AUTUMN'].tail(5).plot.bar(title = 'Higher Autumn Accidents')
    plt.savefig('static/High_Aut_Acc.png')

    plt.subplot(144)
    spring = state_grouped.sort_values('%_SPRING')
    spring['%_SPRING'].tail(5).plot.bar(title = 'Higher Spring Accidents')
    plt.savefig('static/High_Spr_Acc.png')


# ### Let's see the Year wise, which state has met with more number of Accidents

# In[25]:


# Now Let's see Year wise higher accidents of states
highest_accident_states = state_grouped.sort_values('TOTAL', ascending = False)
high_states = list(highest_accident_states.head().index)
df4 = data_monthly.loc[data_monthly['STATE/UT'].isin(high_states),['STATE/UT','YEAR','TOTAL']]

def graph3():
    plt.figure(figsize=(10,6))
    ax = plt.subplot(111)
    for key, grp in df4.groupby(['STATE/UT']):
        ax = grp.plot(ax=ax, kind='line', x='YEAR', y='TOTAL', label=key)
    plt.show()
    plt.savefig('static/Year_wise_Acc.png')


# ### List of top five states met with more number of Accidents

# List of five states met with high accidents
states_list = print(high_states)


# ## Now Let's see the accidents met over a day

## Break up accidents for all states over the time blocks:
#state_time_grouped.info()
def graph4():
    df2=data_time_grouped.sum(axis=0)

    df2.drop(['Total']).T.plot.pie(title='All accidents 2001-2014',subplots=True, figsize=(5,5),autopct='%1.1f%%')

    df2=data_time_grouped.sum(axis=0)
    plt.savefig('static/Acc_in_day')


# ### Now Let's see how Accidents has grown over years

# Let's see how accidents has grown over years
# Notice that accidents are increased exponentially with the years after 2001
def graph5():
    df3 = data_time.groupby(['YEAR']).sum()
    df3.loc[:,'Total'].plot(kind = 'line', title = 'Incresed Accidents over Years')
    plt.savefig('static/Acc_over_years.png')

# States with highest % of accidents in different time blocks
def graph6():
    plt.figure(figsize = (10,5))
    data_time_grouped.sort_values('_%MORNING',ascending = False).head().loc[:,['STATE/UT','_%MORNING']].plot(kind = 'bar', ax = plt.subplot(221),color = 'b')

    data_time_grouped.sort_values('_%AFTER_NOON', ascending = False).head().loc[:,['STATE/UT','_%AFTER_NOON']].plot(kind = 'bar',ax = plt.subplot(222),color = 'r')

    plt.savefig('static/_%Acc_diff_time1.png')

def graph7():
    plt.figure(figsize = (10,5))
    data_time_grouped.sort_values('_%EVENING', ascending = False).head().loc[:,['STATE/UT','_%EVENING']].plot(kind = 'bar',ax = plt.subplot(221),color = 'g')

    data_time_grouped.sort_values('_%NIGHT',ascending = False).head().loc[:,['STATE/UT','_%NIGHT']].plot(kind = 'bar',ax = plt.subplot(222), color = 'y')

    plt.savefig('static/_%Acc_diff_time2.png')

# ## Now Let's see the Accidents met w.r.t Ages

ages = pd.read_csv('Ages.csv', sep = '\t')

ages.head()

ages.drop(axis = 1, columns = ['Age 0-14 - 2014', 'Age 15-24 - 2014', 'Age 25-64 - 2014', 'Age 65 & Above - 2014'],inplace = True)

ages.drop(axis = 1, columns = 'S. No.', inplace = True)

ages.head()

cols = ['State/UT', 'Fatal_18-','Total_18-','Fatal_18-25','Total_18-25','Fatal_25-35','Total_25-35','Fatal_35-45',
       'Total_35-45','Fatal_45-60','Total_45-60','Fatal_60+','Total_60+','No-Age-Fatal','No-Age-Total']

ages.columns = cols

ages.set_index('State/UT',inplace = True)

ages.shape

ages.tail()

ages.drop(index='Total', axis = 0, inplace = True)

ages.tail()

def graph7():
    fig = plt.figure(figsize = (15,3))

    ages.sort_values('Fatal_18-',ascending = False).head().loc[:,['State/UT','Fatal_18-']].plot(kind = 'bar',ax = fig.add_subplot(141), color = 'r')
    
    ages.sort_values('Total_18-',ascending = False).head().loc[:,['State/UT','Total_18-']].plot(kind = 'bar',ax = fig.add_subplot(142), color = 'g')

    ages.sort_values('Fatal_18-25',ascending = False).head().loc[:,['State/UT','Fatal_18-25']].plot(kind = 'bar',ax = fig.add_subplot(143), color = 'b')

    ages.sort_values('Total_18-25',ascending = False).head().loc[:,['State/UT','Total_18-25']].plot(kind = 'bar',ax = fig.add_subplot(144), color = 'y')
                                
    plt.savefig('static/7_a.png')

# In[15]:

def graph8():
    fig = plt.figure(figsize = (15,6))
    ages.sort_values('Fatal_25-35',ascending = False).head().loc[:,['State/UT','Fatal_25-35']].plot(kind = 'bar',ax = fig.add_subplot(241), color = 'y')

    ages.sort_values('Total_25-35',ascending = False).head().loc[:,['State/UT','Total_25-35']].plot(kind = 'bar',ax = fig.add_subplot(242), color = 'b')

    ages.sort_values('Fatal_35-45',ascending = False).head().loc[:,['State/UT','Fatal_35-45']].plot(kind = 'bar',ax = fig.add_subplot(243), color = 'g')

    ages.sort_values('Total_35-45',ascending = False).head().loc[:,['State/UT','Total_35-45']].plot(kind = 'bar',ax = fig.add_subplot(244), color = 'r')

    plt.savefig('static/7_b.png')


def graph9():
    fig = plt.figure(figsize = (15,8))
    ages.sort_values('Fatal_45-60',ascending = False).head().loc[:,['State/UT','Fatal_45-60']].plot(kind = 'bar',ax = fig.add_subplot(341), color = 'y')

    ages.sort_values('Total_45-60',ascending = False).head().loc[:,['State/UT','Total_45-60']].plot(kind = 'bar',ax = fig.add_subplot(342), color = 'b')

    ages.sort_values('Fatal_60+',ascending = False).head().loc[:,['State/UT','Fatal_60+']].plot(kind = 'bar',ax = fig.add_subplot(343), color = 'g')

    ages.sort_values('Total_60+',ascending = False).head().loc[:,['State/UT','Total_60+']].plot(kind = 'bar',ax = fig.add_subplot(344), color = 'r')

    plt.savefig('static/7_c.png')


ages1 = pd.DataFrame(data = ages.index)

ages1.set_index('State/UT',inplace = True)

ages1['%_Age_18-'] = ages['Fatal_18-']/ages['Total_18-']

ages1['%_Age_18-25'] = ages['Fatal_18-25']/ages['Total_18-25']

ages1['%_Age_25-35'] = ages['Fatal_25-35']/ages['Total_25-35']

ages1['%_Age_35-45'] = ages['Fatal_35-45']/ages['Total_35-45']

ages1['%_Age_45-60'] = ages['Fatal_45-60']/ages['Total_45-60']

ages1['%_Age_60+'] = ages['Fatal_60+']/ages['Total_60+']

ages1['%_Age_Unknown'] = ages['No-Age-Fatal']/ages['No-Age-Total']

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

def graph10():
    plt.figure(figsize = (10,8))
    ages1.loc[:,'%_Age_18-':'%_Age_Unknown'].sum(axis=0).plot.pie(title = '% of Accidents in Different Ages',autopct='%1.1f%%')
    plt.savefig('static/ages.png')
