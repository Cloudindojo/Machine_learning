from flask import Flask, render_template, request, redirect, url_for
import tablib
import os
# from hi import main
from random import choice
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('IMDB-Movie-Data.csv')

data.head()

data.shape

data.isna().sum()


rep_Revenue = data['Revenue (Millions)'].median()
rep_Revenue

data['Revenue (Millions)'] = data['Revenue (Millions)'].fillna(rep_Revenue)
data['Revenue (Millions)'].isna().sum()



rep_Meta = data['Metascore'].median()
rep_Meta


data['Metascore'] = data['Metascore'].fillna(rep_Meta)
data['Metascore'].isna().sum()

data.head(10)


movie_yearly_count = data['Year'].value_counts().sort_values(ascending = False)


def graph():
    movie_yearly_count.plot(kind = 'bar', title = 'Number of movies released in a Year')
    plt.xlabel('Years')
    plt.ylabel('Number of Movies')
    plt.savefig('static/samplegraph.png')
    

movies_comparisons = ['Revenue (Millions)', 'Metascore', 'Runtime (Minutes)', 'Votes','Year']

i=0
for comparison in movies_comparisons:
    test = sns.jointplot(x='Rating', y=comparison, data=data, alpha=0.5, color='g', size=8, space=0)
    out='static/testgraph%d.png' %(i)
    test.savefig(out)
    i=i+1

import itertools

unique_genres = data['Genre'].unique()
individual_genres = []
for genre in unique_genres:
    individual_genres.append(genre.split(','))

individual_genres = list(itertools.chain.from_iterable(individual_genres))
individual_genres = set(individual_genres)

individual_genres


print('Number of movies present in each genre')
     
for genre in individual_genres:
    current_genre = data['Genre'].str.contains(genre)

    plt.figure()
    plt.xlabel('Year')
    plt.ylabel('Number of Movies Made')
    plt.title(str(genre))
    plt.savefig('static/genre.png')
    
    

    data[current_genre].Year.value_counts().sort_index().plot(kind = 'bar', color = 'g')
    print(genre, len(current_genre))
   

data['Genre'].value_counts().head()

data['Director'].value_counts().head()


data1 = [data]
data_mapping={'Action,Adventure,Sci-Fi':0,       
'Drama':1,                         
'Comedy,Drama,Romance':2,          
'Comedy':3,                        
'Drama,Romance':4,                 
'Comedy,Drama':5,                  
'Animation,Adventure,Comedy ':6,   
'Action,Adventure,Fantasy':7,      
'Comedy,Romance':8,                
'Crime,Drama,Thriller':9,          
'Crime,Drama,Mystery':10,           
'Action,Adventure,Drama':11,        
'Action,Crime,Drama':12,            
'Horror,Thriller':13,               
'Drama,Thriller':14,                
'Biography,Drama':15,               
'Biography,Drama,History':16,       
'Action,Adventure,Comedy':17,       
'Adventure,Family,Fantasy':18,      
'Action,Crime,Thriller':19,         
'Action,Comedy,Crime':20,           
'Horror':21,                        
'Action,Adventure,Thriller':22,     
'Crime,Drama':23,                   
'Action,Thriller':24,                
'Animation,Action,Adventure':25,     
'Biography,Crime,Drama':26,          
'Thriller':27,                       
'Horror,Mystery,Thriller':28,        
'Biography,Drama,Sport':29}

for dataset in data1:
    dataset['genre'] = dataset['Genre'].map(data_mapping)

dataset['Genre'].head()
def gp():
    ab = data.head()
    print(ab)
    return ab

dataset['genre'].head()
common_value = '0'
data.isna().sum()
data['genre'] = data['genre'].fillna(common_value)
data['genre'].isna().sum()

ab = data.head()

data1=[data]
for dataset in data1:
    dataset['Runtime (Minutes)']=dataset['Runtime (Minutes)'].astype(int)
    dataset.loc[ dataset['Runtime (Minutes)'] <= 66, 'Runtime (Minutes)'] = 1
    dataset.loc[(dataset['Runtime (Minutes)'] > 66) & (dataset['Runtime (Minutes)'] <= 90), 'Runtime (Minutes)'] = 2
    dataset.loc[(dataset['Runtime (Minutes)'] > 90) & (dataset['Runtime (Minutes)'] <= 120), 'Runtime (Minutes)'] = 3
    dataset.loc[(dataset['Runtime (Minutes)'] > 120) & (dataset['Runtime (Minutes)'] <= 140), 'Runtime (Minutes)'] = 4
    dataset.loc[(dataset['Runtime (Minutes)'] > 140) & (dataset['Runtime (Minutes)'] <= 160), 'Runtime (Minutes)'] = 5
    dataset.loc[(dataset['Runtime (Minutes)'] > 160) & (dataset['Runtime (Minutes)'] <= 180), 'Runtime (Minutes)'] = 6
    dataset.loc[dataset['Runtime (Minutes)'] > 180, 'Runtime (Minutes)']=7
data.loc[:,['Runtime (Minutes)','Year','genre','Metascore','Votes']].head()
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X = data.loc[:,['Runtime (Minutes)', 'Year', 'genre', 'Metascore', 'Votes']]
y = data['Rating']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =30)

model_lnr = LinearRegression()

model_lnr.fit(X_train, y_train)

y_pred = model_lnr.predict(X_test)

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)


r2_score(y_test, y_pred)*100

model_lnr.predict([[120, 2017, 13,75, 2575000]])

from sklearn.ensemble import RandomForestRegressor

model_RFR = RandomForestRegressor(n_estimators = 13)

model_RFR.fit(X_train, y_train)

y_pred_RFR = model_RFR.predict(X_test)

r2_score(y_test, y_pred_RFR)* 100

def gp1(a,b,c,d,e):
    from sklearn.model_selection import KFold, cross_val_score

    kfold = KFold(n_splits = 10, shuffle = True, random_state = 0)

    clf = RandomForestRegressor()
    score = cross_val_score(clf, X, y, cv = kfold, n_jobs = 1)
    print(score)

    clf.fit(X_train, y_train)

    s=clf.predict([[a,b,c,d,e]])

    return s

dataset = tablib.Dataset()
with open(os.path.join(os.path.dirname(__file__),'IMDB-Movie-Data.csv')) as f:
    dataset.csv = f.read()

app = Flask(__name__)
app.secret_key = 'super secret key'
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if request.method == 'POST':
        username=str(request.form['username'])
        email=str(request.form['email'])
        password=str(request.form['password'])
        confirm_password=str(request.form['confirm_password'])

        con = psycopg2.connect("host='localhost' dbname='accesion' user='somesh' password='somesh18'")
        cur = con.cursor()
        cur.execute("CREATE TABLE stdregister(username varchar(30),email varchar(30),password varchar(20),confirm_password varchar(20))")
        query =  "INSERT INTO stdregister(username,email,password,confirm_password) VALUES (%s, %s, %s, %s);"
        data = (username,email,password,confirm_password)
        cur.execute(query, data)
        #cur.execute("CREATE TABLE friend1(uname varchar(20),postal_Address varchar(20),personal_Address varchar(20),sex varchar(20),city varchar(20),district varchar(20),state varchar(20),user_name varchar(20),password varchar(20))")
        #cur.execute("INSERT INTO friend1(uname,postal_address,personal_address,sex,city,district,state,user_name,password) VALUES(Name,Postal_Address,Personal_Address,Sex,City,District,State,user_name,password)")
        con.commit()
        if con:
            con.close()
        return redirect(url_for('login'))
    else:
        return render_template('registration.html')


'''try:
    con = psycopg2.connect("host='localhost' dbname='shiva' user='post' password='ashu1234'")
    cur = con.cursor()
    cur.execute("CREATE TABLE friend1(Name varchar(20),Postal_Address varchar(20),Personal_Address varchar(20),Sex varchar(20),City varchar(20),District varchar(20),State varchar(20),user_name varchar(20),password varchar(20))")
    cur.execute("INSERT INTO friend1 VALUES(Name,Postal_Address,Personal_Address,Sex,City,District,State,user_name,password)")
    con.commit()    
except psycopg2.DatabaseError as e:
    if con:
        con.rollback()
     
    print ('Error %s' % e  )  
    sys.exit(1)
finally:
    if con:
        con.close()'''

    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_name1=str(request.form['email'])
        password=str(request.form['password'])
        con = psycopg2.connect("host='localhost' dbname='accesion' user='somesh' password='somesh18'")
        cur = con.cursor()
        
        #cur.execute("SELECT user_name,password FROM register where user_name==user_name1")
        #query =  "SELECT user_name,password FROM register where user_name==%s;"
        data = (user_name1)
        cur.execute("SELECT email, password FROM stdregister where email = %s;",(user_name1,))
        records = cur.fetchall()
        for row in records:
            if request.form['password'] == row[1] and request.form['email'] == row[0]:
                return home()
        return render_template('login.html')

    else:

        return render_template('login.html')


@app.route('/home')
def home():
    graph()
    
    return render_template('home.html')


@app.route('/price', methods=['GET', 'POST'])
def price():
    t=gp()
    a=int(request.form['Runtime (Minutes)'])
    b=int(request.form['Year'])
    c=int(request.form['Genre'])
    d=int(request.form['Votes'])
    e=int(request.form['Metascore'])
    
    dd=gp1(a,b, c, d, e)
    return render_template('hi.html', score = dd, tab=t)

@app.route('/hi')
def graph1():
	return render_template('hi.html')

@app.route('/graphs')
def graphs():
    return render_template('graph.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html', data = data.to_html())

@app.route('/adlogin', methods=['GET','POST'])
def adlogin():
    if request.method=='POST':
        un=str(request.form['username'])
        pd=str(request.form['password'])
        con = psycopg2.connect("host='localhost' dbname='accesion' user='somesh' password='somesh18'")
        cur = con.cursor()
        cur.execute("SELECT username, email FROM stdregister")
        records = cur.fetchall()
        ss=[]
        if un=='admin' and pd == 'password':
            for  i in records:
                ss.append(i)
            return render_template('admin.html', ss=ss)
        return render_template('adminlogin.html')
        
    else:
        return render_template('adminlogin.html')

@app.route('/logout')
def logout():
    return render_template('index.html')


if __name__ == '__main__':
   app.run(debug = True, port=5001)

