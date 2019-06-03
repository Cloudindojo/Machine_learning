from flask import Flask, flash, redirect, render_template, request, session, abort
import time
from datetime import datetime as dt
import os
import sqlite3
import re
import psycopg2
import urllib.parse
import requests
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk import tokenize

app = Flask(__name__)
con=psycopg2.connect("host='localhost' dbname='accesion' user='somesh' password='somesh'")
cur=con.cursor()
urlhistory=[]
def block(url):
	host_path="C:\Windows\System32\drivers\etc\hosts"
	redirect="127.0.0.1"
	website_list = []
	website_list.append(url)
	#while True:
	    
	    # if dt(dt.now().year, dt.now().month,dt.now().day,8)< dt.now() < dt(dt.now().year, dt.now().month, dt.now().day,16):
	    #     print("sorry not allowed.....")
	with open(host_path, 'r+') as file:
	    content = file.read()
	    for website in website_list:
	        if website in content:
	            return "already exist"
	        else:
	            file.write(redirect + " " + website + "\n")
	            return "sucess"
def history1():
	con = sqlite3.connect('C:/Windows.old/Users/acces/Local Settings/Google/Chrome/User Data/Default/History')
	cursor = con.cursor()
	cursor.execute("SELECT url FROM urls")
	urls = cursor.fetchall()
	
	for url in urls:
	    
	    u=str(url)
	    length=len(u)
	    url1=u[2:length-3]
	    
	    aa=re.findall('[b]+[u]+[g]+',url1)
	    """a=str(aa)
	    length1=len(a)
	    a1=a[2:length1-2]"""
	    if(aa==['bug']):
	        print('url is blocked')
	    else:
	    	print(url1)
	    	urlhistory.append(url1)
	return urlhistory

def related(word1):
	tokenizer = RegexpTokenizer('\w+')
	con = sqlite3.connect('C:/Windows.old/Users/acces/Local Settings/Google/Chrome/User Data/Default/History')
	cursor = con.cursor()
	cursor.execute("SELECT url FROM urls")
	urls = cursor.fetchall()
	ss=[]
	t=[]
	for url in urls:
	    
	    u=str(url)
	    length=len(u)
	    url1=u[2:length-3]
	    
	    aa=re.findall('[b]+[u]+[g]+',url1)
	    """a=str(aa)
	    length1=len(a)
	    a1=a[2:length1-2]"""
	    if(aa==['bug']):
	        print('url is blocked')
	    else:
	        ss.append(url1)
	for i in ss:
		tokens = tokenizer.tokenize(i)

		words = []
		   
		for word in tokens:
		    words.append(word.lower())
		    
		sw = nltk.corpus.stopwords.words('english')
		   
		words_ns = []
		    
		for word in words:
		    if word not in sw:
		        words_ns.append(word)
		    # Create freq dist and plot
		if word1 in words_ns:
			t.append(i)
	return t
@app.route('/home' , methods=['GET','POST'])
def home():
	if request.method=='POST':
		url=request.form['url']
		ss={"url":url}
		
		return render_template('home.html', result=result)
	else:
		return render_template('home.html')

@app.route('/history')
def history():
	history1()
	return render_template('history.html', history=urlhistory)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/logout')
def logout():
	return render_template('index.html')

@app.route('/block', methods=['GET','POST'])
def block():
	if request.method=='POST':
		word2=request.form['word']
		result1=related(word2)
		return render_template('block.html', result=result1)
	else:
		return render_template('block.html')
@app.route('/login', methods=['GET','POST'])
def login():
	if request.method=='POST':
		if request.form['username']=='admin' and request.form['password']=='admin':
			return redirect('home')
	return render_template('login.html')

if __name__ == '__main__':
	app.secret_key = os.urandom(12)
	app.run(debug=True)