from flask import Flask, render_template, request, url_for, redirect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('Warm_Up_Predict_Blood_Donations_-_Traning_Data.csv')
df_test = pd.read_csv('Warm_Up_Predict_Blood_Donations_-_Test_Data.csv')

X = df_train.ix[:, 1:5]
y = df_train[['Made Donation in May 2019']]

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_test = df_test.ix[:, 1:5]

X_train = X_train.reset_index(drop = True)
X_cv = X_cv.reset_index(drop = True)
X_test = X_test.reset_index(drop = True)

# We first check for any missing values in the training and testing data.

X_train.isnull().sum()
X_cv.isnull().sum()
X_test.isnull().sum()

# In this case, the color, Red, represents Non-blood Donors, while the color, Blue, represents Donors.
plt.figure(figsize = (20, 10))
plt.subplot(2, 2, 1)
sns.distplot(X_train[y_train.values == 0]['Months since Last Donation'], 
             bins = range(0, 81, 2), color = 'red')
plt.ylabel('Frequency')
plt.title('Distribution of Months Since last Donation for Non-blood Donors')

sns.distplot(X_train[y_train.values == 1]['Months since Last Donation'], 
             bins = range(0, 81, 2), color = 'blue')
plt.ylabel('Frequency')
plt.title('Distribution of Months Since last Donation for Blood Donors')

plt.subplot(2, 2, 2)
sns.distplot(X_train[y_train.values == 0]['Number of Donations'], 
             bins = range(0, 60, 2), color = 'red')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Donations for Non-blood Donors')

sns.distplot(X_train[y_train.values == 1]['Number of Donations'], 
             bins = range(0, 60, 2), color = 'blue')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Donations for Blood Donors')

plt.subplot(2, 2, 3)
sns.distplot(X_train[y_train.values == 0]['Total Volume Donated (c.c.)'], color = 'red')
plt.ylabel('Frequency')
plt.title('Distribution of Total Volume Donated (c.c.) for Non-blood Donors')

sns.distplot(X_train[y_train.values == 1]['Total Volume Donated (c.c.)'], color = 'blue')
plt.ylabel('Frequency')
plt.title('Distribution of Total Volume Donated (c.c.) for Blood Donors')

plt.subplot(2, 2, 4)
sns.distplot(X_train[y_train.values == 0]['Months since First Donation'], color = 'red')
plt.ylabel('Frequency')
plt.title('Distribution of Months since First Donation for Non-blood Donors')

sns.distplot(X_train[y_train.values == 1]['Months since First Donation'], color = 'blue')
plt.ylabel('Frequency')
plt.title('Distribution of Months since First Donation for Blood Donors')
plt.savefig("static/image1.png", dpi=100)
#plt.show()

#We proceed to examine the relationship across the 4 features in the training dataset, by means of a pairsplot.
sns.pairplot(X_train, diag_kind='kde')
plt.savefig("static/image2.png", dpi=100)
#plt.show()

# We check this phenomena using the correlation between these features, by means of a heatmap.
plt.figure(figsize = (20, 10))
X_train_corr = X_train.corr()

sns.heatmap(X_train_corr, annot = True)
plt.savefig("static/image3.png", dpi=100)
#plt.show()


print (set(y_train['Made Donation in May 2019']))
print (set(y_cv['Made Donation in May 2019']))

#Feature Engineering and Feature Selection
X_train['Average Donation per Month'] = (X_train['Total Volume Donated (c.c.)']/ X_train['Months since First Donation'])

# We do the same for the cross validation dataset and the testing dataset.

X_cv['Average Donation per Month'] = X_cv['Total Volume Donated (c.c.)']/X_cv['Months since First Donation']
X_test['Average Donation per Month'] = X_test['Total Volume Donated (c.c.)']/X_test['Months since First Donation']
# Is our new indicator a good predictor for whether the donor donated blood in May 2019?
plt.figure(figsize = (20, 10))

sns.distplot(X_train[y_train.values == 0]['Average Donation per Month'], color = 'red')
plt.ylabel('Frequency')
plt.title('Distribution of Months Since last Donation for Non-blood Donors')

sns.distplot(X_train[y_train.values == 1]['Average Donation per Month'], color = 'blue')
plt.ylabel('Frequency')
plt.title('Distribution of Months Since last Donation for Blood Donors')
plt.savefig("static/image4.png", dpi=100)
#plt.show()

# We include a new variable, average waiting length for donation, to observe the frequency which the donor donates blood.

X_train['Waiting Time'] = ((X_train['Months since First Donation'] - X_train['Months since Last Donation'])
                           /X_train['Number of Donations'])

X_cv['Waiting Time'] = ((X_cv['Months since First Donation'] - X_cv['Months since Last Donation'])
                        /X_cv['Number of Donations'])

X_test['Waiting Time'] = ((X_test['Months since First Donation'] - X_test['Months since Last Donation'])
                          /X_test['Number of Donations'])
# Is this additional feature informative of whether a donor is likely to donate blood in March 2007? Let's find out.
plt.figure(figsize = (20, 10))

sns.distplot(X_train[y_train.values == 0]['Waiting Time'], color = 'red')
plt.ylabel('Frequency')
plt.title('Distribution of Waiting Time for Non-blood Donors')

sns.distplot(X_train[y_train.values == 1]['Waiting Time'], color = 'blue')
plt.ylabel('Frequency')
plt.title('Distribution of Waiting Time for Blood Donors')
plt.savefig("static/image5.png", dpi=100)
#plt.show()


X_train['One-time Donor'] = map(int, (X_train['Number of Donations'] == 1))
tab = pd.crosstab(X_train['One-time Donor'], y_train['Made Donation in May 2019'])

tab.div(tab.sum(1).astype(float), axis=0)

x = y_train.reset_index(drop = True)[X_train['One-time Donor'] == 1]

del X_train['One-time Donor']

X_train['Donated in the past 3-6 months'] = ((X_train['Months since Last Donation'] >= 3) &
                                             (X_train['Months since Last Donation'] <= 6))

X_cv['Donated in the past 3-6 months'] = ((X_cv['Months since Last Donation'] >= 3) &
                                          (X_cv['Months since Last Donation'] <= 6))

X_test['Donated in the past 3-6 months'] = ((X_test['Months since Last Donation'] >= 3) &
                                            (X_test['Months since Last Donation'] <= 6))

X_train['Frequent Donor'] = (X_train['Number of Donations'] >= 5)

X_cv['Frequent Donor'] = (X_cv['Number of Donations'] >= 5)

X_test['Frequent Donor'] = (X_test['Number of Donations'] >= 5)

cols_to_keep = ['Months since Last Donation', 'Number of Donations',
                'Months since First Donation', 'Average Donation per Month', 
                'Waiting Time', 'Donated in the past 3-6 months', 'Frequent Donor']
X_train = X_train[cols_to_keep]; X_cv = X_cv[cols_to_keep]; X_test = X_test[cols_to_keep]


plt.figure(figsize = (20, 10))
train_ = X_train.copy(); train_['y'] = y_train.reset_index(drop = True)
train_corr = train_.corr()

sns.heatmap(train_corr, annot = True)
plt.savefig("static/image6.png", dpi=100)
#plt.show()

plt.figure(figsize = (20, 10))
sns.set(font_scale = 1.5)
(abs(train_corr)
 .y
 .drop('y')
 .sort_values()
 .plot
 .barh())
plt.savefig("static/image7.png", dpi=100)
#plt.show()

from sklearn.preprocessing import StandardScaler

numericFeatures = ['Months since Last Donation', 'Number of Donations', 
                   'Average Donation per Month', 'Waiting Time', 'Months since First Donation']
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[numericFeatures]))
X_cv_scaled = pd.DataFrame(scaler.transform(X_cv[numericFeatures]))
X_test_scaled = pd.DataFrame(scaler.transform(X_test[numericFeatures]))

from sklearn.preprocessing import LabelEncoder

factorVar = ['Frequent Donor', 'Donated in the past 3-6 months']

le = LabelEncoder()

for i in factorVar:
    X_train_scaled[i] = le.fit_transform(X_train[i])
    X_cv_scaled[i] = le.transform(X_cv[i])
    X_test_scaled[i] = le.transform(X_test[i])

# Model 1 - Logistic Regression
from sklearn.linear_model import LogisticRegressionCV

logregr = LogisticRegressionCV(cv = 5, random_state=12, scoring ='neg_log_loss')
logregr = logregr.fit(X_train_scaled, y_train['Made Donation in May 2019'])

y_cv_logregr = logregr.predict_proba(X_cv_scaled)[:, 1]

# Model 2 - RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

randomforest = RandomForestClassifier(random_state = 10)
param_grid = {'n_estimators': [50, 100, 150],
              'max_features': [1, 2, 3]}
rf = GridSearchCV(estimator = randomforest, param_grid = param_grid, cv = 5,
                  scoring = 'neg_log_loss')
rf.fit(X_train_scaled, y_train['Made Donation in May 2019'])

y_cv_rf = rf.predict_proba(X_cv_scaled)[:, 1]


# Model 3 - Support Vector Machine
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm = SVC(probability = True)
param_grid = {'kernel': ['rbf', 'linear'],
              'gamma': np.logspace(-4, 1, 10),
              'random_state': [10]}
svm = GridSearchCV(svm, param_grid = param_grid, cv = 5,
                   scoring = 'neg_log_loss')

svm.fit(X_train_scaled, y_train['Made Donation in May 2019'])

y_cv_svm = svm.predict_proba(X_cv_scaled)[:, 1]

from sklearn.metrics import log_loss

# Logistic Regression
print ('Logistic Regression - Entropy Loss: ', log_loss(y_cv, y_cv_logregr))

# Random Forest
print ('Random Forest - Entropy Loss: ', log_loss(y_cv, y_cv_rf))


# Support Vector Machine
print ('Support Vector Machine - Entropy Loss: ', log_loss(y_cv, y_cv_svm))


X_total = pd.concat([pd.DataFrame(X_train_scaled), pd.DataFrame(X_cv_scaled)])
y_total = pd.concat([y_train, y_cv]).values

final_model = logregr.fit(X_total, y_total)
y_test = final_model.predict_proba(X_test_scaled)[:, 1]

print (y_test.mean(), df_train['Made Donation in May 2019'].values.mean())

submission = pd.read_csv('Warm_Up_Predict_Blood_Donations_-_Submission_Format.csv', index_col = 0)
submission['Made Donation in May 2019'] = y_test

submission.to_csv('test_submission.csv')

app = Flask(__name__)


@app.route('/')
def index():
  return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        un=str(request.form['username'])
        pd=str(request.form['password'])
        if un=='admin' and pd == 'password':
            return home()
        return render_template('login.html')
        
    else:
        return render_template('login.html')

@app.route('/logout')
def logout():
 
  return render_template('login.html')



@app.route('/home')
def home():
	return render_template('home.html')
@app.route('/train')
def train():
  return render_template('train.html', s1=df_train.to_html())

@app.route('/test')
def test():
  return render_template('test.html', s1=df_test.to_html())

@app.route('/analysis')
def analysis():
  return render_template('analysis.html')

@app.route('/featureanalysis')
def featureanalysis():
  return render_template('featureanalysis.html')

@app.route('/prediction')
def prediction():
  return render_template('prediction.html', s1=submission.to_html())



if __name__ == '__main__':
    app.run(debug=True)