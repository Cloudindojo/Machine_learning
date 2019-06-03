from flask import Flask, render_template,  request, redirect, url_for, send_from_directory, flash
import os
import sys
from pymongo import MongoClient
from analyser import review_analyser 
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import keras as kr
import sklearn
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import itertools
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('data1.csv')

data.set_index('Date', inplace = True)
data1 = data.transpose()
dates = pd.date_range(start = '2019-03-15', freq = 'D', periods = len(data1.columns)*1)
data_np = data1.transpose().as_matrix()
shape = data_np.shape
data_np = data_np.reshape((shape[0] * shape[1], 1))
df = pd.DataFrame({'Mean' : data_np[:,0]})
df.set_index(dates, inplace = True)

plt.figure(figsize = (15,5))
plt.plot(df.index, df['Mean'])
plt.title('daily Mean')
plt.xlabel('Date')
plt.ylabel('Mean across Day')
plt.savefig('static/graph.png')

dataset = df.values
train = dataset[0:15,:]
test = dataset[15:,:]
print("Original data shape:",dataset.shape)
print("Train shape:",train.shape)
print("Test shape:",test.shape)

# Converting the data into MinMax Scaler because to avoid any outliers present in our dataset
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data.shape


x_train, y_train = [], []
for i in range(5,len(train)):
    x_train.append(scaled_data[i-5:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

 # Creating and fitting the model

model = Sequential()
model.add(LSTM(units = 6, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(units = 6))
model.add(Dense(1))

model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(x_train, y_train, epochs=10, batch_size = 1, verbose = 2)
# Now Let's perform same operations that are done on train set
inputs = df[len(df) - len(test) - 5:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(5,inputs.shape[0]):
    X_test.append(inputs[i-5:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
Mean = model.predict(X_test)
Mean1 = scaler.inverse_transform(Mean)
#plotting the train, test and forecast data
train = df[:15]
test = df[15:]
test['Predictions'] = Mean1
trainpred = model.predict(X_test,steps=2)
#x_train shape
x_train.shape
pred = scaler.inverse_transform(trainpred)
pred[0:24] 
testScore = math.sqrt(mean_squared_error(test['Mean'], trainpred[:6,0]))*100
print('Accuracy Score: %.2f' % (testScore))

dates1 = pd.date_range(start = '2019-03-30', freq = 'D', end = '2019-04-10')

new_df = pd.DataFrame({'Predicted_values':pred[:,0]})

new_df.set_index(dates1, inplace = True)







currentDT = datetime.now()


time = currentDT.strftime("%I:%M:%S %p")

app = Flask(__name__)
date=currentDT.strftime("%Y-%m-%d")
#date=datetime.now().today()
#time=datetime.now().time()
connection = MongoClient("localhost", 27017)
db = connection.mydatabase 
mla=db.mla
mp=db.mp
user=db.user
meetings=db.meetings
comments1=db.usercomments
app.secret_key = 'super'
@app.route('/')
def index():
   return render_template('login_form.html')

@app.route('/login', methods=['GET','POST'])
def login():
	if request.method == 'POST':
		un = str(request.form['UserName'])
		p = str(request.form['Password'])
		radio=str(request.form['contact'])
		if radio == "admin":
			if un=="admin" and p=="admin":
				adu=[]
				adml=[]
				admp=[]
				for i in user.find():
					adu.append(i)
				for j in mla.find():
					if j['designation'] == 'mla':
						adml.append(j)
					else:
						admp.append(j)
				li=[]
				
				#paste here val
				if comments1.count() == 0 :
					stars = 0
				else :
					avg=0
					b=0
					for rev in comments1.find():
						avg = avg + rev['rating']
						b = b + 1
					stars = avg / b
				val = stars
				comments = 	comments1.find()	
				flash(comments)	
				return render_template("home.html", user1=adu, mla=adml, mp=admp, val = val, ss=new_df.to_html())
			else:
				return render_template('login_form.html', data="authentication failed")
		elif radio == "mla":
			for i in mla.find():
				if i['designation'] == 'mla':
					if un==i["username"] and p==i["password"]:
						ml=[]
						for i in user.find():
							ml.append(i)
						return render_template("mla.html", ml=ml)
					
			return render_template('login_form.html', data="authentication failed to mla")
		elif radio == "mp":
			for i in mla.find():
				if i['designation'] == 'mp':
					if un==i["username"] and p==i["password"]:
						ml=[]
						for i in user.find():
							ml.append(i)
						return render_template("mp.html")
				
			return render_template('login_form.html', data="authentication failed to mp")
		elif radio == "user":
			for i in user.find():
				met=[]
				if un==i["UserName"] and p==i["Password"]:
					for i in meetings.find():
						met.append(i)
					return render_template("user.html", ll=met)
				
			return render_template('login_form.html', data="authentication failed")
		else:
			return render_template('login_form.html', data="select raido button")
	else:
		return render_template("login_form.html")
@app.route('/mlampreg', methods=['GET','POST'])
def mlampreg():
	if request.method == 'POST':
		radio=str(request.form['contact1'])
		firstName=str(request.form['First_Name'])
		lastName=str(request.form['Last_Name'])
		dob=str(request.form.get('Birthday_day'))+"/"+str(request.form.get('Birthday_Month'))+"/"+str(request.form.get('Birthday_Year'))
		email=str(request.form['Email_Id'])
		phno=str(request.form['Mobile_Number'])
		Gender=str(request.form['Gender'])
		username=str(request.form['username'])
		password=str(request.form['password'])
		address=str(request.form['Address'])
		consistency=str(request.form['consistency'])
		city=str(request.form['City'])
		pincode=str(request.form['Pin_Code'])
		state=str(request.form['State'])
		country=str(request.form['Country'])
		mladata={"designation":radio, "firstName":firstName,"lastName":lastName,"dob":dob,"email":email,"phno":phno,"Gender":Gender,"username":username,"password":password,"address":address,"consistency":consistency,"city":city,"pincode":pincode,"state":state,"country":country}
		mla.insert_one(mladata)
		return render_template('mlampreg.html', da="sucess")
	else:
		return render_template('mlampreg.html')

@app.route('/register', methods=['GET','POST'])
def register():
	if request.method == 'POST':
		un = request.form['UserName']
		fn = request.form['FirstName']
		ln = request.form['LastName']
		d = request.form['Date']
		g = request.form['Gmail']
		p = request.form['Password']
		cp = request.form['ConfirmPassword']
		m =  request.form['MobileNumber']
		r = request.form['gridRadios']
		a = request.form['Address']
		cs = request.form['Consistency']
		c = request.form['City']
		pc = request.form['Pincode']
		st = request.form['State']
		ct = request.form['Country']
		
		userdata = {'date':date, 'time':time ,'UserName': un, 'FirstName': fn, 'LastName':ln, 'Date':d, 'Gmail':g, 'Password': cp, 'Gender': r, 'Address':a, 'Consistency':cs, 'City':c, 'Pincode':pc, 
		'State':st, 'Country':ct}
		user.insert_one(userdata)
		return redirect('login')
	return render_template('Registration_form.html')

@app.route('/meetingdetails', methods=['GET','POST'])
def meetingdetails():
	if request.method == 'POST':
		place=request.form['place']
		date=request.form['date']
		time=request.form['time']
		purpose=request.form['purpose']
		meetingdetails1={'place':place, 'date':date, 'time':time, 'purpose':purpose}
		meetings.insert_one(meetingdetails1)
		return render_template('meetingdetails.html', data='sucess')
	else:
		return render_template('meetingdetails.html')
	

@app.route('/meets', methods=['GET','POST'])
def meets():
	ls=[]
	for i in meetings.find():
		ls.append(i)
	if request.method == 'POST':
		comment=request.form['comment']
		a,b,c=review_analyser(comment)
		
		comments1.insert_one({"comment":comment,"rating":a,"star":b,"category":c})

		return render_template('meatings.html', meetings1=ls)
	else:
		return render_template('meatings.html', meetings1=ls)


@app.route('/logout', methods=['GET','POST'])
def logout():
	return render_template('login_form.html')

if __name__ == '__main__':
   app.run(debug = True)