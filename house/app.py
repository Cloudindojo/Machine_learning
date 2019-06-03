from flask import Flask, render_template, request, url_for, redirect, session
from tkinter import *
import csv
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy.polynomial.polynomial as poly
import psycopg2
import sys
app = Flask(__name__)
con = psycopg2.connect("host='localhost' dbname='accesion' user='somesh' password='somesh18'")
cur = con.cursor()

def click(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10) :


    with open('predict.csv','w',newline='') as f :
        dataentry=csv.writer(f,delimiter=",")
        dataentry.writerow(('bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','condition','grade','sqft_above','sqft_basement'))
        dataentry.writerow((d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,))

    data = pd.read_csv('housedata.csv')
    print(data.head())
    print(data.describe())
    target=pd.read_csv('housetarget.csv')
    print(target.head())
    print(target.describe())
    predict=pd.read_csv('predict.csv')

    features = data.iloc[:,:-1]
    print(features)
    labels = data.iloc[:, -1:]
    

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(features)
    print(X_poly)

    

    poly_x_train, poly_x_test, poly_y_train, poly_y_test = train_test_split(X_poly, labels, test_size=0.1, random_state=2)

    poly_reg = LinearRegression()
    poly_reg.fit(poly_x_train, poly_y_train)

    score=poly_reg.score(poly_x_test, poly_y_test)
    print(score)

    print(predict)
    predict_output = poly_features.fit_transform(predict)
    pre=poly_reg.predict(predict_output)
    print(pre)

    result=pre
    

    def updateGraph():
        y_predict = poly_reg.predict(poly_x_test)
        plt.plot(poly_x_test, poly_y_test, "r.", label="Test Data")
        plt.plot(poly_x_test, y_predict, "g.", label="Predicted Data")
        plt.xlim(0,200000)
        # plt.ylim(1000000, 4000000)
        plt.title('Predicted(y) vs Test Data(x)')
        plt.savefig('static/img.png')
    updateGraph()
    return result
    #     sns.despine()
    #     fig.canvas.draw()

    # fig = plt.figure(1)
    # # Special type of "canvas" to allow for matplotlib graphing
    # canvas = FigureCanvasTkAgg(fig, master=window)
    # plot_widget = canvas.get_tk_widget()

    # # Add the plot to the tkinter widget
    # plot_widget.grid(row=1, column=2, rowspan=14)


app.secret_key = 'super secret key'
@app.route('/home', methods=['GET','POST'])
def home():
	if request.method=='POST':
		BedRooms=request.form['BedRooms']
		BathRooms=request.form['BathRooms']
		Squarefeet=request.form['Squarefeet']
		SquareLot=request.form['SquareLot']
		Floors=request.form['Floors']
		WaterFronts=request.form['WaterFronts']
		SwimmingPool=request.form['SwimmingPool']
		Grade1=request.form['Grade']
		if Grade1=='a+':
			Grade=1
		elif Grade1=='a':
			Grade=2
		else:
			Grade=3
		SquareFeetAbove=request.form['SquareFeetAbove']
		SquareFeetBasement=request.form['SquareFeetBasement']
		ss=click(BedRooms,BathRooms,Squarefeet,SquareLot,Floors,WaterFronts,SwimmingPool,Grade,SquareFeetAbove,SquareFeetBasement)
		return render_template('result.html',BedRooms=BedRooms,BathRooms=BathRooms,Squarefeet=Squarefeet,SquareLot=SquareLot,Floors=Floors,WaterFronts=WaterFronts,SwimmingPool=SwimmingPool,Grade=Grade,SquareFeetAbove=SquareFeetAbove,SquareFeetBasement=SquareFeetBasement,ss=ss)
	else:
		return render_template('home.html')

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        un=str(request.form['username'])
        pd=str(request.form['password'])

        cur.execute("SELECT user_name, password FROM housereg where user_name = %s;",(un,))
        records = cur.fetchall()
        for i in records:
            if un==i[0] and pd == i[1]:
                return redirect('home')
        return redirect('login')
        
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


@app.route('/adlogin', methods=['GET','POST'])
def adlogin():
    if request.method=='POST':
        un=str(request.form['username'])
        pd=str(request.form['password'])

        cur.execute("SELECT user_name, uname FROM housereg")
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
    return redirect('index')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')
	

if __name__ == "__main__":
	app.run(debug=True)


# user_name=request.form['userName']
#         password=request.form['password']
#         Name=request.form['name']
#         cur.execute("CREATE TABLE housereg(uname varchar(20),user_name varchar(20),password varchar(20))")
#         query =  "INSERT INTO housereg(uname,user_name,password) VALUES (%s, %s, %s);"
#         data = (Name,user_name,password)
#         cur.execute(query, data)
#         con.commit()
#         if con:
#             con.close()

# cur.execute("SELECT user_name, password FROM housereg where user_name = %s;",(un,))
#         records = cur.fetchall()
#         for i in records:
#             if un=='user' and pd == 'password':
#                 return redirect('home')