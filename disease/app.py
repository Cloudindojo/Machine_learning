from flask import Flask, render_template, flash, request, url_for, redirect
app = Flask(__name__)

from tkinter import *
import numpy as np
import pandas as pd
import psycopg2

con = psycopg2.connect("host='localhost' dbname='accesion' user='somesh' password='somesh18'")
cur = con.cursor()

# from gui_stuff import *

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

# print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
t3=[]
# ------------------------------------------------------------------------------------------------------

def DecisionTree(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5):

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]

    for k in range(0,len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t3.clear()
        t3.append(disease[a])
    else:
        t3.clear()
        t3.append("Not Found")
    return t3[0]


def randomforest(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5):
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.clear()
        t3.append(disease[a])
    else:
        t3.clear()
        t3.append("Not Found")
    return t3[0]

def NaiveBayes(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5):
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1,Symptom2,Symptom3,Symptom4,Symptom5]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.clear()
        t3.append(disease[a])
    else:
        t3.clear()
        t3.append("Not Found")
    return t3[0]

@app.route("/home", methods=['GET', 'POST'])
def home():
	if request.method=='POST':
            name=str(request.form['name'])
            Symptom1=str(request.form['Symptom1'])
            Symptom2=str(request.form['Symptom2'])
            Symptom3=str(request.form['Symptom3'])
            Symptom4=str(request.form['Symptom4'])
            Symptom5=str(request.form['Symptom5'])
            result1=str(DecisionTree(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5))
            result2=str(randomforest(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5))
            result3=str(NaiveBayes(Symptom1,Symptom2,Symptom3,Symptom4,Symptom5))
            #cur.execute("CREATE TABLE disease(uname varchar(20),Symptom1 varchar(20),Symptom2 varchar(20),Symptom3 varchar(20),Symptom4 varchar(20),Symptom5 varchar(20), DecisionTree varchar(20), randomforest varchar(20), NaiveBayes varchar(20))")
            query =  "INSERT INTO disease(uname, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, DecisionTree, randomforest, NaiveBayes) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"
            data = (name, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, result1, result2, result3)
            cur.execute(query, data)
            con.commit()
            return render_template('result.html',name=name,Symptom1=Symptom1,Symptom2=Symptom2,Symptom3=Symptom3,Symptom4=Symptom4,Symptom5=Symptom5,result1=result1,result2=result2,result3=result3)
	else:
		return render_template('home.html', ll=l1)

@app.route('/', methods=['GET','POST'])
def login():
    if request.method=='POST':
        un=str(request.form['username'])
        pd=str(request.form['password'])

        cur.execute("SELECT user_name, password FROM stockreg where user_name = %s;",(un,))
        records = cur.fetchall()
        for i in records:
            if un==i[0] and pd == i[1]:
                return redirect('home')
        return redirect('/')
        
    else:
        return render_template('login.html')

@app.route('/reg', methods=['GET','POST'])
def reg():
    if request.method=='POST':
        un=str(request.form['username'])
        pd=str(request.form['password'])
        Name=str(request.form['name'])
        #cur.execute("CREATE TABLE stockreg(uname varchar(20),user_name varchar(20),password varchar(20))")
        query =  "INSERT INTO stockreg(uname,user_name,password) VALUES (%s, %s, %s);"
        data = (Name,un,pd)
        cur.execute(query, data)
        con.commit()
        return redirect('/')
        
    else:
        return render_template('reg.html')


@app.route('/adlogin', methods=['GET','POST'])
def adlogin():
    if request.method=='POST':
        un=str(request.form['username'])
        pd=str(request.form['password'])

        cur.execute("SELECT * FROM disease")
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
    return render_template('login.html')
@app.route('/logout1')
def logout1():
    return render_template('adminlogin.html')

if __name__ == "__main__":
    app.run(debug=True)
